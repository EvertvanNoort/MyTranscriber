# sudo apt-get install wkhtmltopdf
import pdfkit

def convert_html_to_pdf(html_file_path, output_pdf_path):
    css = '/home/evert/Desktop/audio/style.css'
    pdfkit.from_file(html_file_path, output_pdf_path, options={"enable-local-file-access": ""})

# Replace these with your file paths
html_file_path = 'output.html'
output_pdf_path = 'output.pdf'  

convert_html_to_pdf(html_file_path, output_pdf_path)
