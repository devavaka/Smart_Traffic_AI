from fpdf import FPDF
from utils.database import get_all_logs
from utils.helpers import get_timestamp

def generate_pdf_report(save_path, current_stats=None, current_risk=None, plates_list=None):
    logs = get_all_logs()
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Traffic Monitoring Analytics Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Generated: {get_timestamp()}", ln=True, align='C')
    pdf.ln(10)
    
    # Current Stats Section if available
    if current_stats:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Current Session Summary", ln=True, align='L')
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 8, txt=f"Total Vehicles: {current_stats.get('total', 0)}", ln=True)
        pdf.cell(100, 8, txt=f"Vehicles Per Minute (VPM): {current_stats.get('vpm', 0)}", ln=True)
        pdf.cell(100, 8, txt=f"Current Accident Risk: {current_risk:.1f}%", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt="Vehicle Breakdown:", ln=True)
        pdf.set_font("Arial", '', 12)
        for vtype, count in current_stats.get("types", {}).items():
            pdf.cell(100, 8, txt=f"- {vtype}: {count}", ln=True)
        pdf.ln(10)
    
    # Detected Plates Section
    if plates_list:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Detected Number Plates", ln=True, align='L')
        pdf.set_font("Arial", '', 11)
        # Limit to last 30 plates to prevent crazy long PDFs from plates alone
        for plt in plates_list[:30]: 
            # FPDF (standard fonts) only supports latin-1, so remove unrenderable characters like '│'
            safe_plt_str = str(plt).replace('│', '|').replace('🚦', '').replace('🚗', '').encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(200, 7, txt=safe_plt_str, ln=True)
        pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Recent Database Logs", ln=True, align='L')
    
    if not logs:
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt="No historical database analytics available yet.", ln=True, align='L')
        try:
            pdf.output(save_path)
            return True
        except Exception:
            return False

    # A simple table of recent records
    # Column widths
    w = [40, 20, 20, 15, 15, 15, 15, 20]
    
    # Header
    pdf.set_font("Arial", 'B', 10)
    headers = ["Time", "Total", "VPM", "Car", "Bike", "Truck", "Bus", "Risk %"]
    for i in range(len(headers)):
        pdf.cell(w[i], 10, headers[i], border=1, align='C')
    pdf.ln()
    
    # Data
    pdf.set_font("Arial", '', 9)
    for row in logs:
        # row: id, time(1), total(2), vpm(3), car(4), bike(5), truck(6), bus(7), risk(8)
        pdf.cell(w[0], 8, str(row[1])[:19], border=1) # Time
        pdf.cell(w[1], 8, str(row[2]), border=1, align='C') # Total
        pdf.cell(w[2], 8, f"{float(row[3]):.1f}", border=1, align='C') # VPM
        pdf.cell(w[3], 8, str(row[4]), border=1, align='C') # Car
        pdf.cell(w[4], 8, str(row[5]), border=1, align='C') # Bike
        pdf.cell(w[5], 8, str(row[6]), border=1, align='C') # Truck
        pdf.cell(w[6], 8, str(row[7]), border=1, align='C') # Bus
        pdf.cell(w[7], 8, f"{float(row[8]):.1f}%", border=1, align='C') # Risk
        pdf.ln()
        
    try:
        # Attempt to save, handle encoding errors by encoding and decoding
        pdf.output(save_path)
        return True
    except Exception as e:
        print(f"Error exporting PDF: {e}")
        # fpdf might fail on some chars, so we can set a fallback handling or ignore
        return False
