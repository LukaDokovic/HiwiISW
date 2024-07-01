import re

def modify_gcode(input_file, output_file):
    # Header und Footer
    header = """#SLOPE[TYPE=STEP]
#SCALE X0.5 Y0.5
#SCALE ON
$FOR P1=0, 3, 1
M180=9900
M181=100
G0 F50000
G1 F1000
"""
    footer = """G0 X0.00 Y0.00
$ENDFOR
#SCALE OFF
M30
"""

    # Funktion zum Überprüfen, ob eine Zeile eine Z-Achsen-Bewegung ist
    def is_z_axis_movement(line):
        return re.match(r'G[01] Z', line)

    # Funktion zum Hinzufügen von Laser-Ein/Aus-Befehlen
    def add_laser_control(lines):
        modified_lines = []
        laser_on = False
        for line in lines:
            if 'G1' in line and not laser_on:
                modified_lines.append('M150\n')  # Laser ein
                laser_on = True
            if 'G0' in line and laser_on:
                modified_lines.append('M151\n')  # Laser aus
                laser_on = False
            modified_lines.append(line)
        if laser_on:  # Falls der Laser am Ende noch an ist, schalten wir ihn aus
            modified_lines.append('M151\n')
        return modified_lines

    # Original-G-Code einlesen
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Unnötige Zeilen entfernen und Z-Achsen-Bewegungen herausfiltern
    filtered_lines = [line for line in lines if not is_z_axis_movement(line)]

    # Lasersteuerungsbefehle hinzufügen
    final_lines = add_laser_control(filtered_lines)

    # Modifizierten G-Code in eine neue Datei schreiben
    with open(output_file, 'w') as file:
        file.write(header)
        file.writelines(final_lines)
        file.write(footer)

# Beispielaufruf der Funktion
modify_gcode('input_gcode.txt', 'output_gcode.txt')
