import subprocess
import json
import os

def generate_config():
    # 1. Run Terraform to get IPs in JSON format
    try:
        print("Reading IPs from Terraform...")
        # The command is 'terraform output -json ips_alumnos'
        result = subprocess.run(
            ["terraform", "output", "-json", "ips_alumnos"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        ips = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running Terraform: {e}")
        return
    except json.JSONDecodeError:
        print("Error: Terraform did not return valid JSON. Did you run 'terraform apply'?")
        return

    # 2. Print the block for copy/paste
    print("\n" + "="*50)
    print("COPY FROM HERE AND PASTE INTO THE SHARED DOC")
    print("="*50 + "\n")

    print("INSTRUCTIONS FOR THE STUDENT:")
    print("1. Copy YOUR assigned block (find your name in the table).")
    print("2. Paste it into your VS Code SSH configuration file.")
    print("3. IMPORTANT: Change the 'IdentityFile' line to the path where you saved YOUR .pem key")
    print("-" * 50 + "\n")

    # 3. Iterate over the IPs and generate each student block
    for i, ip in enumerate(ips):
        student_num = i + 1
        
        print(f"--- STUDENT {student_num} (Assigned to: __________) ---")
        print(f"Host seminario-mlops")
        print(f"    HostName {ip}")
        print(f"    User ubuntu")
        # Use a generic Windows example path (most common setup)
        print(f"    IdentityFile \"C:\\Users\\<YOUR_USER>\\Downloads\\mlops_seminary_ssh.pem\"")
        
        # Optional: prevents the "Are you sure?" prompt on connection
        print(f"    StrictHostKeyChecking no") 
        print("") # Blank line separator

if __name__ == "__main__":
    generate_config()
