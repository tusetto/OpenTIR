#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenTIR Project Setup Script
Genera da zero l'intera struttura del progetto OpenTIR con tutti i file aggiornati.
Eseguire questo file per resettare o ricostruire il progetto.
"""

import os
import sys
import shutil
import stat

# --- CONFIGURAZIONE PROGETTO ---
PROJECT_NAME = "opentir"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Contenuto dei file principali
FILES_CONTENT = {
    "opentir/__init__.py": """
__version__ = "1.0.0"
__author__ = "OpenTIR Team"
""",

    "opentir/profiles.py": """
import numpy as np

def conic_sag(r, c, k):
    \"\"\"Calcola la sagitta di una superficie conica.\"\"\"
    if k == -1:
        return (c * r**2) / (1 + np.sqrt(1 - (1 + k) * c**2 * r**2 + 1e-15))
    term = 1 - (1 + k) * c**2 * r**2
    term = np.maximum(term, 0)  # Evita radici negative
    return (c * r**2) / (1 + np.sqrt(term))

def build_conic_profile(z_vals, c, k, diameter):
    \"\"\"Genera profili conici.\"\"\"
    r = np.linspace(0, diameter/2, len(z_vals))
    z = conic_sag(r, c, k)
    return r, z

def build_freeform_profile(z_vals, r_control, z_control):
    \"\"\"Genera profili freeform tramite interpolazione.\"\"\"
    # Implementazione semplificata per demo
    return r_control, z_control

def profile_to_surfaces(profile_data, surface_type='conic'):
    \"\"\"Converte un profilo in una lista di superfici ottiche.\"\"\"
    surfaces = []
    # Logica di conversione
    surfaces.append({'type': surface_type, 'data': profile_data})
    return surfaces
""",

    "opentir/raytrace.py": """
import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(direction)

class Surface:
    def __init__(self, z_func, r_max, material='glass'):
        self.z_func = z_func
        self.r_max = r_max
        self.material = material

    def intersect(self, ray):
        # Calcolo intersezione semplificato
        return np.array([0, 0, 0]), True

def trace_rays(rays, surfaces):
    \"\"\"Traccia un insieme di raggi attraverso le superfici.\"\"\"
    paths = []
    for ray in rays:
        current_ray = ray
        path = [current_ray.origin]
        for surf in surfaces:
            pt, hit = surf.intersect(current_ray)
            if hit:
                path.append(pt)
                # Qui si calcolerebbe la rifrazione/riflessione
        paths.append(path)
    return paths
""",

    "opentir/fresnel.py": """
import numpy as np

def calculate_fresnel_reflection(n1, n2, theta_i):
    \"\"\"Calcola i coefficienti di Fresnel.\"\"\"
    # Legge di Snell
    sin_t = (n1 / n2) * np.sin(theta_i)
    if np.abs(sin_t) > 1:
        return 1.0, 1.0  # Riflessione totale
    
    cos_t = np.sqrt(1 - sin_t**2)
    cos_i = np.cos(theta_i)
    
    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))**2
    rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i))**2
    
    return (rs + rp) / 2, 1 - (rs + rp) / 2
""",

    "opentir/gui.py": """
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import locali corretti
from .profiles import profile_to_surfaces, build_conic_profile
from .raytrace import Ray, Surface, trace_rays

class OpticalElement:
    def __init__(self, name, elem_type, params):
        self.name = name
        self.type = elem_type
        self.params = params

class OpenTIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenTIR - Progettazione Ottica")
        
        self.elements = []
        self.source_params = {'type': 'LED', 'angle': 120, 'rays': 100}
        
        self.setup_ui()
        self.update_element_tree()

    def setup_ui(self):
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Pannello sinistro (Controlli)
        control_frame = ttk.LabelFrame(main_frame, text="Strumenti", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=5)
        
        # Pulsanti Superfici
        ttk.Label(control_frame, text="Aggiungi Superficie:").grid(row=0, column=0, pady=5)
        ttk.Button(control_frame, text="Lente Conica", command=lambda: self.add_element('lens_conic')).grid(row=1, column=0, pady=2, sticky=tk.EW)
        ttk.Button(control_frame, text="Superficie Freeform", command=lambda: self.add_element('surface_freeform')).grid(row=2, column=0, pady=2, sticky=tk.EW)
        ttk.Button(control_frame, text="Riflettore", command=lambda: self.add_element('reflector')).grid(row=3, column=0, pady=2, sticky=tk.EW)
        
        # SEZIONE SORGENTE (Spostata qui sotto le superfici)
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, sticky='ew', pady=10)
        ttk.Label(control_frame, text="Sorgente:", font=('TkDefaultFont', 0, 'bold')).grid(row=5, column=0, pady=5)
        ttk.Button(control_frame, text="Configura Sorgente LED", command=self.edit_source).grid(row=6, column=0, pady=2, sticky=tk.EW)

        # Albero Elementi
        tree_frame = ttk.LabelFrame(main_frame, text="Elementi Ottici", padding="5")
        tree_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5)
        
        columns = ('nome', 'tipo')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='treeheadings', height=15)
        self.tree.heading('#0', text='Nome Elemento')
        self.tree.heading('nome', text='Dettagli')
        self.tree.heading('tipo', text='Tipo')
        
        self.tree.column('#0', width=200)
        self.tree.column('nome', width=150)
        self.tree.column('tipo', width=100)
        
        self.tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Toolbar Albero
        tree_toolbar = ttk.Frame(tree_frame)
        tree_toolbar.grid(row=1, column=0, sticky=(tk.E, tk.W), pady=5)
        ttk.Button(tree_toolbar, text="Modifica", command=self.edit_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(tree_toolbar, text="Elimina", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        
        # Area Grafico
        plot_frame = ttk.LabelFrame(main_frame, text="Simulazione", padding="5")
        plot_frame.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        ttk.Button(plot_frame, text="Aggiorna Simulazione", command=self.run_simulation).grid(row=1, column=0, pady=5)

        # Configurazione griglia
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=2)
        main_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

    def update_element_tree(self):
        # Pulisci albero
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Inserisci Sorgente come primo elemento (fisso)
        source_id = self.tree.insert("", "end", text=f"💡 Sorgente LED ({self.source_params['angle']}°)", values=("Angolo: {}°, Raggi: {}".format(self.source_params['angle'], self.source_params['rays']), "Sorgente"))
        self.tree.item(source_id, tags=('source',))
        
        # Inserisci altri elementi
        for idx, elem in enumerate(self.elements):
            icon = "🔵" if 'lens' in elem.type else ("🟣" if 'freeform' in elem.type else "🪞")
            name = f"{icon} {elem.name}"
            self.tree.insert("", "end", text=name, values=(str(elem.params), elem.type), tags=(elem.type,))

    def add_element(self, elem_type):
        name = f"Elem_{len(self.elements)+1}"
        params = {'curvature': 0.01, 'material': 'BK7'}
        self.elements.append(OpticalElement(name, elem_type, params))
        self.update_element_tree()

    def edit_source(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Configura Sorgente")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Angolo di emissione (gradi):").pack(pady=5)
        angle_var = tk.StringVar(value=str(self.source_params['angle']))
        ttk.Entry(dialog, textvariable=angle_var).pack(pady=5)
        
        ttk.Label(dialog, text="Numero Raggi:").pack(pady=5)
        rays_var = tk.StringVar(value=str(self.source_params['rays']))
        ttk.Entry(dialog, textvariable=rays_var).pack(pady=5)
        
        def save():
            try:
                self.source_params['angle'] = float(angle_var.get())
                self.source_params['rays'] = int(rays_var.get())
                self.update_element_tree()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Errore", "Inserisci valori numerici validi")
        
        ttk.Button(dialog, text="Salva", command=save).pack(pady=20)

    def edit_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        item = self.tree.item(selection[0])
        tags = item['tags']
        
        if 'source' in tags:
            self.edit_source()
        else:
            messagebox.showinfo("Info", "Modifica elemento ottico (da implementare)")

    def delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        item = self.tree.item(selection[0])
        if 'source' in item['tags']:
            messagebox.showwarning("Attenzione", "La sorgente non può essere eliminata.")
            return
            
        # Trova indice e rimuovi
        # Logica semplificata per demo
        messagebox.showinfo("Info", "Elemento eliminato (logica da completare)")

    def run_simulation(self):
        self.ax.clear()
        self.ax.set_title("Simulazione Raggi")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Z (mm)")
        
        # Disegna sorgente
        self.ax.plot(0, 0, 'y*', markersize=15, label='Sorgente')
        
        # Disegna elementi dummy
        z_vals = np.linspace(-10, 10, 100)
        for elem in self.elements:
            if 'lens' in elem.type:
                self.ax.plot(z_vals, np.sqrt(100 - z_vals**2), 'b-', alpha=0.5)
        
        self.ax.legend()
        self.ax.set_aspect('equal')
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = OpenTIRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
""",

    "run_gui.py": """
#!/usr/bin/env python3
import sys
import os

# Assicura che la root directory sia nel path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from opentir.gui import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Errore di importazione: {e}")
    print("Assicurati di aver eseguito Setup.py per generare la struttura corretta.")
    sys.exit(1)
""",

    "run_cli.py": """
#!/usr/bin/env python3
print("CLI OpenTIR - In sviluppo")
""",

    "requirements.txt": """
numpy>=1.20.0
matplotlib>=3.4.0
scipy>=1.7.0
"""
}

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clean_directory(root):
    print("🧹 Pulizia directory...")
    for item in os.listdir(root):
        if item in ['__pycache__', '.git'] or item.endswith('.pyc'):
            path = os.path.join(root, item)
            if os.path.isdir(path):
                shutil.rmtree(path, onerror=remove_readonly)
            else:
                os.remove(path)
    print("✅ Pulizia completata.")

def create_structure(root):
    print("📂 Creazione struttura cartelle...")
    dirs = [
        os.path.join(root, PROJECT_NAME),
        os.path.join(root, "examples"),
        os.path.join(root, "tests")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Cartelle create.")

def write_files(root):
    print("📝 Scrittura file Python...")
    for filepath, content in FILES_CONTENT.items():
        full_path = os.path.join(root, filepath)
        # Assicurati che la cartella padre esista
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"   - Creato: {filepath}")
    print("✅ File scritti.")

def main():
    print(f"🚀 Avvio Setup OpenTIR in: {ROOT_DIR}")
    
    if not os.path.exists(ROOT_DIR):
        print("❌ Directory radice non trovata.")
        return

    clean_directory(ROOT_DIR)
    create_structure(ROOT_DIR)
    write_files(ROOT_DIR)
    
    print("\n✨ Setup completato con successo!")
    print("Puoi ora eseguire il programma con:")
    print("   python run_gui.py")

if __name__ == "__main__":
    main()