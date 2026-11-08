"""Create sample_docs/nimbus_annual_report.pdf: a 6-page report with charts, a table and a diagram.

It is the companion of employee_handbook.pdf for trying image-aware answers and chat memory.
Every page has the Nimbus logo in the header (it must be ignored, not indexed). The pages:

    page 1  Overview            text only       (no figure is needed to answer)
    page 2  Revenue by quarter  text + bar chart
    page 3  Revenue by region   text + table
    page 4  Headcount           text + line chart
    page 5  Organisation        text + an embedded diagram image
    page 6  Outlook             text only

Questions to try, and where the answer is:

    "What was the total revenue in 2025?"                -> page 1 or 2, no image
    "How did revenue change through the year?"           -> page 2, shows the bar chart
      then, as a follow-up: "And which quarter was the best?"   (memory: "the year" is 2025)
    "Which region earned the most?"                      -> page 3, the table
    "How fast did the team grow?"                        -> page 4, shows the line chart
    "Show me how the company is organised."              -> page 5, shows the diagram
      then: "How many people work in the largest one?"          (memory: largest department)
    "What are the plans for 2026?"                       -> page 6, no image
    "What is the CEO's favourite colour?"                -> not in the document

Run from the project root:
    python scripts/make_sample_report_pdf.py
"""
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image as PlatypusImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "sample_docs" / "nimbus_annual_report.pdf"

NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E76F51")

# The numbers all agree with each other (quarters, regions and departments add up).
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
REVENUE = [4.2, 5.1, 6.3, 8.0]                       # USD millions, total 23.6
REGIONS = [("North America", 11.8, "18%"), ("Europe", 7.1, "22%"), ("Asia-Pacific", 4.7, "31%")]
HEADCOUNT = [120, 135, 150, 172]
DEPARTMENTS = [("Engineering", 78), ("Sales", 41), ("Support", 33), ("People & Finance", 20)]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                  # very old Pillow
        return ImageFont.load_default()


def logo_png() -> bytes:
    """A small header logo, drawn here so the script needs no image files."""
    img = Image.new("RGB", (360, 110), (31, 58, 95))
    d = ImageDraw.Draw(img)
    d.ellipse((24, 22, 90, 88), fill=(42, 157, 143))
    d.text((112, 28), "NIMBUS", fill="white", font=_font(46))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def org_diagram_png() -> bytes:
    """An organisation chart as a picture (an embedded image, not vector drawing)."""
    w, h = 1400, 760
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    big, small = _font(38), _font(30)

    def box(cx, cy, bw, bh, fill, lines):
        d.rounded_rectangle((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2),
                            radius=14, fill=fill, outline=(31, 58, 95), width=3)
        total = len(lines) * 40
        for i, (text, font) in enumerate(lines):
            tw = d.textlength(text, font=font)
            d.text((cx - tw / 2, cy - total / 2 + i * 42), text, fill="white", font=font)

    d.text((w / 2 - d.textlength("Nimbus Software: organisation chart, December 2025", font=small) / 2,
            20), "Nimbus Software: organisation chart, December 2025", fill=(31, 58, 95),
           font=small)
    box(w // 2, 190, 420, 120, (31, 58, 95), [("Chief Executive", big), ("172 people in total", small)])
    xs = [190, 520, 880, 1210]
    for x in xs:
        d.line((w // 2, 250, w // 2, 330), fill=(31, 58, 95), width=4)
        d.line((w // 2, 330, x, 330), fill=(31, 58, 95), width=4)
        d.line((x, 330, x, 440), fill=(31, 58, 95), width=4)
    for x, (name, count) in zip(xs, DEPARTMENTS):
        box(x, 520, 300, 160, (42, 157, 143), [(name, small), (f"{count} people", big)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def revenue_chart() -> Drawing:
    d = Drawing(430, 250)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 55, 40, 340, 170
    chart.data = [REVENUE]
    chart.categoryAxis.categoryNames = [f"{q} 2025" for q in QUARTERS]
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 10, 2
    chart.bars[0].fillColor = TEAL
    chart.barLabelFormat = "%.1f"
    chart.barLabels.nudge = 8
    chart.barLabels.fontSize = 10
    chart.categoryAxis.labels.fontSize = 10
    chart.valueAxis.labels.fontSize = 10
    d.add(chart)
    d.add(Label(), name="title")
    d.title.setOrigin(225, 228)
    d.title.setText("Revenue per quarter, 2025 (USD millions)")
    d.title.fontSize = 12
    return d


def headcount_chart() -> Drawing:
    d = Drawing(430, 250)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 55, 40, 340, 170
    plot.data = [list(zip(range(1, 5), HEADCOUNT))]
    plot.lines[0].strokeColor = ORANGE
    plot.lines[0].strokeWidth = 2.5
    plot.lines[0].symbol = None
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueSteps = 1, 4, [1, 2, 3, 4]
    plot.xValueAxis.labelTextFormat = lambda v: f"{QUARTERS[int(v) - 1]} 2025"
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueStep = 100, 180, 20
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 10
    d.add(plot)
    legend = Legend()
    legend.x, legend.y, legend.fontSize = 70, 228, 10
    legend.colorNamePairs = [(ORANGE, "Employees at the end of the quarter")]
    d.add(legend)
    return d


def _styles():
    base = getSampleStyleSheet()
    return {
        "h1": base["Heading1"],
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=11, leading=16, spaceAfter=10),
        "caption": ParagraphStyle("Caption", parent=base["Italic"], fontSize=9, textColor=colors.grey,
                                  spaceAfter=10),
    }


def _header_footer(logo: bytes):
    reader = ImageReader(io.BytesIO(logo))

    def draw(canvas, doc):
        canvas.saveState()
        canvas.drawImage(reader, 2.5 * cm, A4[1] - 1.9 * cm, width=3.0 * cm, height=0.92 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 2.5 * cm, A4[1] - 1.5 * cm,
                               "Annual Report 2025 | Confidential example")
        canvas.drawString(2.5 * cm, 1.3 * cm, "Nimbus Software Ltd. (a fictional company)")
        canvas.drawRightString(A4[0] - 2.5 * cm, 1.3 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def build(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    total = sum(REVENUE)
    story = []

    def page(title, paragraphs, *flowables, last=False):
        story.append(Paragraph(title, s["h1"]))
        story.append(Spacer(1, 6))
        for text in paragraphs:
            story.append(Paragraph(text, s["body"]))
        story.extend(flowables)
        if not last:
            story.append(PageBreak())

    page("Overview", [
        "Nimbus Software Ltd. closed 2025 with total revenue of 23.6 million USD, up 24% on "
        "2024. The company ended the year with 172 employees and 340 business customers.",
        "The head office is in Colombo, with a second office in Berlin. This report covers "
        "revenue, headcount, the organisation and the plans for 2026.",
        "Net profit for the year was 3.1 million USD, a margin of 13%. The company has no debt.",
    ])

    page("Revenue by quarter", [
        "Revenue grew in every quarter of 2025. It rose from 4.2 million USD in Q1 to 8.0 "
        "million USD in Q4, so the fourth quarter was almost twice as large as the first.",
        "Q4 was the strongest quarter, helped by the launch of the Nimbus Analytics product in "
        f"October. The four quarters add up to {total:.1f} million USD.",
        ], revenue_chart(),
        Paragraph("Figure 1. Quarterly revenue in 2025.", s["caption"]))

    region_rows = [["Region", "Revenue 2025 (USD millions)", "Growth vs 2024"]] + [
        [name, f"{rev:.1f}", growth] for name, rev, growth in REGIONS] + [
        ["Total", f"{sum(r for _, r, _ in REGIONS):.1f}", "24%"]]
    table = Table(region_rows, colWidths=[5 * cm, 6 * cm, 4 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6EF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    page("Revenue by region", [
        "North America is the largest market and earned half of all revenue. Asia-Pacific is "
        "the smallest region but grew the fastest, at 31% compared with 2024.",
        ], table, Spacer(1, 8),
        Paragraph("Table 1. Revenue and growth by region.", s["caption"]))

    page("Headcount", [
        "The team grew from 120 people at the end of Q1 to 172 people at the end of Q4, an "
        "increase of 52 people. Most new hires joined Engineering.",
        "Hiring was fastest in the second half of the year, after the Analytics launch.",
        ], headcount_chart(),
        Paragraph("Figure 2. Employees at the end of each quarter, 2025.", s["caption"]))

    diagram = PlatypusImage(io.BytesIO(org_diagram_png()), width=15.5 * cm, height=15.5 * cm * 760 / 1400)
    page("Organisation", [
        "Nimbus is organised into four departments. Engineering is the largest, with 78 "
        "people. Sales has 41 people, Support 33 and People & Finance 20.",
        "The Chief Executive leads all four departments directly.",
        ], diagram, Spacer(1, 4),
        Paragraph("Figure 3. Organisation chart, December 2025.", s["caption"]))

    page("Outlook for 2026", [
        "In 2026 Nimbus plans to open an office in Singapore to serve Asia-Pacific customers, "
        "and to grow the team to about 210 people.",
        "The revenue target for 2026 is 30 million USD. The company will also release version "
        "2 of Nimbus Analytics in the third quarter.",
        "The main risks are slower hiring in Engineering and currency changes in Europe.",
    ], last=True)

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, title="Nimbus Software Annual Report 2025",
        leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=3 * cm, bottomMargin=2.5 * cm,
    )
    draw = _header_footer(logo_png())
    doc.build(story, onFirstPage=draw, onLaterPages=draw)
    return output


if __name__ == "__main__":
    path = build()
    print(f"Created {path.relative_to(ROOT)} with 6 pages")
