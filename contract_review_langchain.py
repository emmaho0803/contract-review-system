# contract_review_langchain.py
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import mimetypes
from dotenv import load_dotenv
import PyPDF2
from docx import Document

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class LangChainContractReviewSystem:
    def __init__(self, model_name="gpt-3.5-turbo"):
        load_dotenv()
        self.llm = ChatOpenAI(model_name=model_name)
        self.prompt = PromptTemplate(
            input_variables=["clause"],
            template="""
你是一位專業合約審查員。請簡要分析以下合約條款，指出潛在風險和需要注意的要點：
{clause}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

        self.local_resources = {
            'keywords': {
                'high_risk': self._load_keywords('high_risk.txt'),
                'business_terms': self._load_keywords('business_terms.txt')
            }
        }

        os.makedirs("cached_analyses", exist_ok=True)

    def _load_keywords(self, filename: str) -> List[str]:
        try:
            with open(f"local_resources/keywords/{filename}", 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def _extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_text_from_word(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def handle_uploaded_file(self, file_path: str) -> Tuple[bool, str]:
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type == "application/pdf":
                text = self._extract_text_from_pdf(file_path)
            elif mime_type in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                text = self._extract_text_from_word(file_path)
            else:
                return False, f"不支持的文件格式 ({mime_type})，請上傳 PDF 或 Word 文檔"
            return True, text
        except Exception as e:
            return False, f"文件處理失敗: {str(e)}"

    def split_into_clauses(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        sections = re.split(r'第\s*\d+\s*[條|款|節]', text)
        return [s.strip() for s in sections if s.strip()]

    def classify_clause(self, text: str) -> Tuple[str, float]:
        score = 0.0
        for kw in self.local_resources['keywords']['high_risk']:
            if kw in text:
                score += 0.1
        for kw in self.local_resources['keywords']['business_terms']:
            if kw in text:
                score += 0.05
        if score >= 0.5:
            return "高風險條款", min(score, 1.0)
        elif score >= 0.3:
            return "商業條款", min(score, 1.0)
        else:
            return "一般條款", min(score, 1.0)

    def _load_cached_analysis(self, clause: str) -> Optional[Dict[str, str]]:
        cache_path = Path("cached_analyses") / (str(hash(clause)) + ".json")
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_cached_analysis(self, clause: str, result: Dict[str, str]):
        cache_path = Path("cached_analyses") / (str(hash(clause)) + ".json")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def analyze_clauses(self, clauses: List[str]) -> List[Dict[str, str]]:
        results = []
        for clause in clauses:
            cached = self._load_cached_analysis(clause)
            if cached:
                results.append(cached)
                continue

            clause_type, priority = self.classify_clause(clause)
            high_risk = [kw for kw in self.local_resources['keywords']['high_risk'] if kw in clause]

            if priority < 0.5:
                result_data = {
                    "clause": clause,
                    "analysis": f"本地分析：未發現明顯風險。({clause_type})",
                    "risk_keywords": high_risk,
                    "source": "local"
                }
            else:
                gpt_result = self.chain.run({"clause": clause})
                result_data = {
                    "clause": clause,
                    "analysis": gpt_result,
                    "risk_keywords": high_risk,
                    "source": "gpt"
                }

            self._save_cached_analysis(clause, result_data)
            results.append(result_data)
        return results

    def generate_report(self, results: List[Dict[str, str]]) -> str:
        report_lines = ["合約審查報告", "=" * 30]
        for i, r in enumerate(results, 1):
            report_lines.append(f"第 {i} 條款｜契約內容：{r['clause'][:60]}...")
            report_lines.append(f"分析方式：{r['source']}")
            report_lines.append(f"分析結果：{r['analysis'].strip()}")
            if r['risk_keywords']:
                report_lines.append(f"⚠️ 發現風險關鍵詞：{', '.join(r['risk_keywords'])}")
            report_lines.append("─" * 40)
        return "\n".join(report_lines)

    def review_contract_file(self, file_path: str) -> str:
        success, content_or_error = self.handle_uploaded_file(file_path)
        if not success:
            return content_or_error

        clauses = self.split_into_clauses(content_or_error)
        results = self.analyze_clauses(clauses)
        return self.generate_report(results)

if __name__ == "__main__":
    import gradio as gr

    system = LangChainContractReviewSystem()

    def gradio_review(file):
        with open(file.name, "rb") as f:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
        return system.review_contract_file(tmp_path)

    iface = gr.Interface(
        fn=gradio_review,
        inputs=gr.File(label="📄 上傳合約文件 (.pdf / .docx)", file_types=[".pdf", ".docx"]),
        outputs=gr.Textbox(label="📝 合約審查報告", lines=30, show_copy_button=True),
        title="合約審查系統",
        description="上傳合約，系統將自動分析每條條款並給出風險建議"
    )
    iface.launch()