import argparse
import requests

SERVER_URL = "http://localhost:5000/getEmbeddings"

def upload_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        files = {'file': pdf_file}
        response = requests.post(SERVER_URL, files=files)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"Response content: {response.content}")
            raise
        return response.json()

def main():
    parser = argparse.ArgumentParser(description="Upload a PDF file to the server for processing.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    response = upload_pdf(args.pdf_path)
    print(response)

if __name__ == "__main__":
    main()