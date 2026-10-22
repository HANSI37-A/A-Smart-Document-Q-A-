"""Create sample_docs/employee_handbook.pdf: a 6-page handbook for a made-up company.

One topic per page, so you always know which page holds each answer:
    page 1  Welcome            page 4  Probation and Performance
    page 2  Leave Policy       page 5  IT Security
    page 3  Remote Work        page 6  Expenses

Run from the project root:
    python scripts/make_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "sample_docs" / "employee_handbook.pdf"

PAGES = [
    ("Welcome to Nimbus Software", [
        "Nimbus Software Ltd. is a fictional software company used as an example in this "
        "project. This handbook explains the policies that apply to all full-time employees.",
        "Office hours are 9:00 am to 5:30 pm, Monday to Friday. Core collaboration hours, "
        "when everyone should be available for meetings, are 10:00 am to 3:00 pm.",
        "Questions about this handbook should be sent to the People Team at "
        "people@nimbus.example.",
    ]),
    ("Leave Policy", [
        "Full-time employees receive 14 days of paid annual leave per calendar year. Annual "
        "leave must be requested in the HR portal at least 7 days in advance.",
        "Employees also receive 7 days of paid sick leave per year. A medical certificate is "
        "required for sick leave longer than 2 consecutive days.",
        "Up to 5 days of unused annual leave can be carried over to the next year. Carried-over "
        "days must be used before 31 March, after which they expire.",
        "Parental leave is 84 working days for the primary caregiver and 10 working days for "
        "the secondary caregiver.",
    ]),
    ("Remote Work", [
        "Employees may work from home up to 2 days per week after completing probation. "
        "Remote days must be agreed with your line manager in advance.",
        "When working remotely, you must be reachable during the core collaboration hours and "
        "must use the company VPN to access internal systems.",
    ]),
    ("Probation and Performance", [
        "New employees serve a probation period of 6 months. During probation, you meet your "
        "line manager every month to review your progress.",
        "During probation, either side may end the employment with 1 week of written notice. "
        "After probation, the notice period is 1 month.",
        "Formal performance reviews take place twice a year, in June and December.",
    ]),
    ("IT Security", [
        "Passwords must be at least 14 characters long. Multi-factor authentication is "
        "required for all company accounts.",
        "Laptops are replaced every 3 years. Lost or stolen devices must be reported to the "
        "IT Helpdesk within 24 hours.",
    ]),
    ("Expenses", [
        "Business expenses are reimbursed within 10 working days of submitting a claim. Claims "
        "must be submitted in the expenses portal, with receipts, within 30 days of the purchase.",
        "Meals during business travel are reimbursed up to 40 USD per day. Flights must be "
        "booked in economy class. Any single purchase above 200 USD needs your line manager's "
        "approval before you buy it.",
    ]),
]


def build(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=10)

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, title="Nimbus Software Employee Handbook",
        leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 8))
        for text in paragraphs:
            story.append(Paragraph(text, body))
        if i < len(PAGES) - 1:
            story.append(PageBreak())  # one topic per page, so page numbers are predictable
    doc.build(story)
    return output


if __name__ == "__main__":
    path = build()
    print(f"Created {path.relative_to(ROOT)} with {len(PAGES)} pages")
