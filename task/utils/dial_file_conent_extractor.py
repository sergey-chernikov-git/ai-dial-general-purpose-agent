import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        self.dial_client = Dial(api_key=api_key, base_url=endpoint)

    def extract_text(self, file_url: str) -> str:
        file = self.dial_client.files.download(file_url)
        return self.__extract_text(
            file_content=file.get_content().decode("utf-8"),
            file_extension=Path(file.filename).suffix.lower(),
            filename=file.filename
        )

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        try:
            if file_extension == ".txt":
                return file_content.decode(
                    "utf-8",
                    errors="ignore"
                )

            if file_extension == ".pdf":
                b_content = io.BytesIO(file_content)
                pdf_file = pdfplumber.open(b_content)
                page_texts = []
                for page in pdf_file.pages:
                    page_texts.append(page.extract_text())
                return "\n".join(page_texts)

            if file_extension == ".csv":
                txt_data = file_content.decode('utf-8', errors='ignore')
                buffer = io.StringIO(txt_data)
                dataframe = pd.read_csv(buffer)
                return dataframe.to_markdown()

            if file_extension in ['.html', '.htm']:
                html_data = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_data, features="html.parser")
                for elem in soup(["script", "style"]):
                    elem.decompose()
                return soup.get_text(separator="\n", strip=True)

            return file_content.decode(
                "utf-8",
                errors="ignore"
            )
        except Exception as e:
            print(e)
            return ""
