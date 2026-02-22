import subprocess
import json
import os
import csv
import unicodedata

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFRA_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(SCRIPT_DIR, "students", "students_dl.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "seminar_access.html")
# ---------------------

def read_students():
    if not os.path.exists(INPUT_FILE):
        print(f"⚠️  WARNING: '{INPUT_FILE}' not found. Generating generic names.")
        return []

    students = []
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            first_name = (row.get("Nombre") or "").strip()
            last_name = (row.get("Apellido(s)") or "").strip()
            full_name = " ".join(part for part in [first_name, last_name] if part)
            if full_name:
                students.append({
                    "name": full_name,
                    "first_name": first_name,
                    "last_name": last_name
                })

    def normalize_for_sort(text):
        # Sort names accent-insensitively, e.g. "Álvaro" as "Alvaro"
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()

    students.sort(
        key=lambda s: (
            normalize_for_sort(s["first_name"]),
            normalize_for_sort(s["last_name"]),
            normalize_for_sort(s["name"])
        )
    )
    return [s["name"] for s in students]

def generate_html():
    # 1. Get IPs from Terraform
    try:
        print("🌍 Reading IPs from Terraform...")
        result = subprocess.run(
            ["terraform", "output", "-json", "ips_alumnos"], 
            cwd=INFRA_DIR,
            capture_output=True, 
            text=True, 
            check=True
        )
        ips = json.loads(result.stdout)
    except Exception as e:
        print(f"❌ Error running Terraform: {e}")
        return

    # 2. Read students list
    students = read_students()
    
    # 3. HTML Content Construction
    # NOTE: Double curly braces {{ }} are used for CSS so Python doesn't confuse them with variables
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MLOps Seminar Access</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; color: #333; }}
            h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #007bff; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            
            pre {{ 
                margin: 0; 
                font-family: Consolas, "Courier New", monospace; 
                background-color: #f4f4f4; 
                padding: 10px; 
                border-radius: 4px;
                border: 1px solid #ccc;
                white-space: pre; 
            }}
            .instructions {{ background: #e7f3fe; padding: 15px; border-left: 6px solid #2196F3; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="instructions">
            <h3>🎓 Student Instructions</h3>
            <p>Search for your name, and locate your assigned SSH configuration cell. You will see that the command is already pre-populated with all the necessary information.</p>
            <p>The SSH configuration is the block that you have to copy into the VSCode SSH config file.</p>
        </div>

        <h2>MLOps Seminar - Access Credentials</h2>

        <table>
            <thead>
                <tr>
                    <th style="width: 20%">Student Name</th>
                    <th style="width: 15%">Assigned IP</th>
                    <th>SSH Configuration (Copy the block below)</th>
                </tr>
            </thead>
            <tbody>
    """

    # 4. Generate rows dynamically
    max_count = max(len(ips), len(students))
    
    for i in range(max_count):
        # Get IP or placeholder
        ip = ips[i] if i < len(ips) else "WAITING FOR DEPLOY"
        
        # Get Name or placeholder
        name = students[i] if i < len(students) else f"RESERVE SEAT {i+1}"
        
        # Create the SSH block securely using explicit newlines
        # This avoids indentation errors in Python
        ssh_block = (
            f"Host mlops-seminar\n"
            f"    HostName {ip}\n"
            f"    User ubuntu\n"
            f"    IdentityFile \"C:\\Users\\eps\\Downloads\\mlops_seminary_ssh.pem\"\n"
            f"    StrictHostKeyChecking no"
        )

        row = f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{ip}</td>
                <td>
                    <pre>{ssh_block}</pre>
                </td>
            </tr>
        """
        html_content += row

    # 5. Close HTML tags
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # 6. Write the file to disk
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Success! File generated: {os.path.abspath(OUTPUT_FILE)}")
    print("👉 Open this file in your browser, copy the table, and paste it into Google Docs.")

if __name__ == "__main__":
    generate_html()
