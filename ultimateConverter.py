import os
import re

def modify_gcode(input_file, output_file):
    # Header und Footer
    header = """#SLOPE[TYPE=STEP]
#SCALE X0.05 Y0.05
#SCALE ON
$FOR P1=0, 100, 1
M180=9900
M181=100
G0 F50000
G1 F1000
"""
    footer = """$ENDFOR
#SCALE OFF
M30
"""

    # Funktion zum Entfernen von unerwünschten Zeilen
    def is_unwanted_line(line):
        return (re.match(r'\(.*\)', line) or
                line.startswith('%') or
                line.startswith('M3') or
                line.startswith('G21') or
                line.startswith('M5') or
                line.startswith('M2') or
                line.strip() == "G00 Z5.000000")

    # Funktion zum Entfernen von Z-, F-Koordinaten und Kommentaren in Klammern bei G01-Befehlen und ganzer Zeilen, die nur eine Z-Bewegung enthalten
    def clean_line(line):
        # Entferne Zeilen, die nur eine Z-Bewegung enthalten
        if re.match(r'G[01] Z[-\d.]+$', line):
            return None
        # Entferne Kommentare in Klammern bei G01-Befehlen
        line = re.sub(r'\(.*?\)', '', line)
        # Entferne nur die Z- und F-Koordinate aus Zeilen, die auch X- und Y-Koordinaten enthalten
        line = re.sub(r' Z[-\d.]+', '', line)
        line = re.sub(r' F[-\d.]+', '', line)
        return line

    # Funktion zum Anwenden des Skalierungsfaktors auf X und Y Koordinaten
    def apply_scaling(line):
        match = re.match(r'(G[01]) X([-\d.]+) Y([-\d.]+)', line)
        if match:
            command, x, y = match.groups()
            scaled_x = float(x) * 0.05
            scaled_y = float(y) * 0.05
            return f"{command} X{scaled_x:.6f} Y{scaled_y:.6f}\n"
        return line

    # Original-G-Code einlesen
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Unnötige Zeilen herausfiltern
    filtered_lines = [line for line in lines if not is_unwanted_line(line)]

    # Z- und F-Koordinaten und Kommentare entfernen oder Zeilen löschen und Skalierung anwenden
    processed_lines = []
    for line in filtered_lines:
        modified_line = clean_line(line)
        if modified_line:
            processed_lines.append(apply_scaling(modified_line))
    
    # Lasersteuerungsbefehle hinzufügen
    final_lines = []
    laser_on = False
    for line in processed_lines:
        if 'G00' in line:
            if laser_on:
                final_lines.append('M151\n')  # Laser aus
                laser_on = False
            final_lines.append(line)
            final_lines.append('M150\n')  # Laser ein
        elif 'G01' in line:
            final_lines.append(line)
            laser_on = True
        else:
            final_lines.append(line)

    if laser_on:  # Falls der Laser am Ende noch an ist, schalten wir ihn aus
        final_lines.append('M151\n')

    # Entfernen von leeren G01-Zeilen
    final_lines = [line for line in final_lines if not (line.startswith('G01') and line.strip() == 'G01')]

    # Entfernen von doppelten oder überflüssigen G00 X0.0000 Y0.0000
    cleaned_lines = []
    for i, line in enumerate(final_lines):
        if line == "G00 X0.0000 Y0.0000\n":
            if i == 0 or final_lines[i - 1] != line:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Entfernen von M150 nach dem letzten M151
    if cleaned_lines and cleaned_lines[-1] == 'M150\n':
        cleaned_lines.pop()

    # Entfernen von doppelten G00 X0.0000 Y0.0000 am Ende
    if len(cleaned_lines) > 1 and cleaned_lines[-1] == "G00 X0.0000 Y0.0000\n" and cleaned_lines[-2] == "G00 X0.0000 Y0.0000\n":
        cleaned_lines.pop()

    # Modifizierten G-Code in eine neue Datei schreiben
    with open(output_file, 'w') as file:
        file.write(header)
        file.writelines(cleaned_lines)
        file.write(footer)

def convert_and_modify_gcode(input_ngc_file, output_nc_file):
    # Temporäre .txt-Datei für die Modifikation
    temp_txt_file = 'temp_gcode.txt'
    
    # Konvertiere .ngc zu .txt
    with open(input_ngc_file, 'r') as ngc_file:
        lines = ngc_file.readlines()

    with open(temp_txt_file, 'w') as txt_file:
        txt_file.writelines(lines)

    # Modifiziere den G-Code und speichere ihn als .nc-Datei
    modify_gcode(temp_txt_file, output_nc_file)
    
    # Entferne die temporäre .txt-Datei
    os.remove(temp_txt_file)

    print(f"Datei '{input_ngc_file}' wurde erfolgreich in '{output_nc_file}' umgewandelt und modifiziert.")

def main():
    while True:
        input_file = input("Bitte geben Sie den Namen der .ngc-Datei ein: ")
        if os.path.isfile(input_file):
            output_file = os.path.splitext(input_file)[0] + '_converted.nc'
            convert_and_modify_gcode(input_file, output_file)
            break
        else:
            print("Datei nicht gefunden. Bitte geben Sie einen gültigen Dateinamen ein.")

if __name__ == "__main__":
    main()
