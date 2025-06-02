import tkinter as tk
import csv
from tkinter import Frame, Label, Entry, Button, BooleanVar, Scrollbar, Text, END, LEFT, RIGHT, BOTH, Y
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.traveling_salesman_problem.SalesmanProblemGA import SalesmanProblemGA, multipopulation_evolution
from src.data.data_prep import load_location_data, get_route_points

def load_coordinates_from_csv(filepath):
    """
    Lê um CSV com formato: <nome do local>;<latitude>, <longitude>
    Retorna um dicionário {nome: "lat,lng"}.
    """
    coord_dict = {}
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if len(row) >= 2:
                name = row[0].strip()
                coord = row[1].replace(" ", "").strip()  # Remove espaços entre lat e lng
                coord_dict[name] = coord
    return coord_dict

def generate_google_maps_url(route_names, coord_dict):
    """
    Gera uma URL do Google Maps com base na sequência de nomes da rota e no dicionário de coordenadas.
    """
    coords = [coord_dict[name] for name in route_names if name in coord_dict]
    if not coords:
        return "Coordenadas não encontradas para gerar link."
    base_url = "https://www.google.com/maps/dir/"
    # Google Maps: 1 origem + até 23 waypoints + 1 destino = 25 pontos
    max_points = 15
    urls = []
    for i in range(0, len(coords), max_points - 1):
        chunk = coords[i:i + max_points]
        urls.append(base_url + "/".join(chunk))
    return urls

class InterfaceGA:

    def __init__(self, root):
        """
        Interface gráfica para configurar e executar o Algoritmo Genético multipopulacional (duas ilhas)
        com atualização de gráfico somente no momento de migração.
        """
        self.root = root
        self.root.title("TSP Genetic Algorithm - Multipopulação")
        self.root.geometry("900x700")
        self.root.configure(bg="#2E2E2E")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_frames()
        self.create_form()
        self.create_result_area()

        # Variáveis de controle da execução
        self.islands = []
        self.migration_interval = 0
        self.total_generations = 0
        self.running = False
        self.thread = None
        self.best_route_final = None
        self.best_fitness_final = None

    def create_frames(self):
        # Frame esquerdo para o formulário
        self.frame_left = Frame(self.root, bg="#2E2E2E", width=300)
        self.frame_left.pack(side=LEFT, fill=Y, padx=10, pady=10)

        # Frame direito para exibir resultados e gráfico
        self.frame_right = Frame(self.root, bg="#2E2E2E")
        self.frame_right.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

    def create_form(self):
        Label(self.frame_left, text="Configuração (válida para ambas as ilhas)", bg="#2E2E2E", fg="white",
              font=("Verdana", 14, "bold")).pack(anchor="w", pady=(0,5))
        self.entries = {}
        fields = [
            ("Population Size:", "pop_size"),
            ("Crossover Rate (0-1):", "cx_rate"),
            ("Mutation Rate (0-1):", "mu_rate"),
            ("Generations:", "gens"),
            ("Elitism Size:", "elitism"),
            ("Tournament Size:", "tourn"),
        ]
        for label_text, var_name in fields:
            Label(self.frame_left, text=label_text, bg="#2E2E2E", fg="white",
                  font=("Verdana", 11)).pack(anchor="w", pady=3)
            entry = Entry(self.frame_left, bg="#555555", fg="white", insertbackground="white",
                          font=("Verdana", 11))
            entry.pack(fill="x", pady=2)
            self.entries[var_name] = entry

        self.tournament_var = BooleanVar(value=True)
        tk.Checkbutton(self.frame_left, text="Use Tournament Selection", variable=self.tournament_var,
                       bg="#2E2E2E", fg="white", selectcolor="#2E2E2E",
                       font=("Verdana", 11)).pack(anchor="w", pady=5)

        Label(self.frame_left, text=" ", bg="#2E2E2E").pack(pady=5)

        Label(self.frame_left, text="Parâmetros Globais", bg="#2E2E2E", fg="white",
              font=("Verdana", 14, "bold")).pack(anchor="w", pady=(0,5))
        # Migration interval
        Label(self.frame_left, text="Migration Interval (gerações):", bg="#2E2E2E", fg="white",
              font=("Verdana", 11)).pack(anchor="w", pady=3)
        self.entry_mig = Entry(self.frame_left, bg="#555555", fg="white", insertbackground="white",
                               font=("Verdana", 11))
        self.entry_mig.pack(fill="x", pady=2)

        Label(self.frame_left, text="Locations File Path:", bg="#2E2E2E", fg="white",
              font=("Verdana", 11)).pack(anchor="w", pady=3)
        self.entry_path = Entry(self.frame_left, bg="#555555", fg="white", insertbackground="white",
                                font=("Verdana", 11))
        self.entry_path.insert(0, "src/data/locations.xlsx")
        self.entry_path.pack(fill="x", pady=2)

        # Botão para executar
        self.run_button = Button(self.frame_left, text="Run", bg="#555555", fg="white",
                                 font=("Verdana", 12, "bold"), command=self.on_run)
        self.run_button.pack(pady=15)

        # Label para status
        self.status_label = Label(self.frame_left, text="Status: Aguardando", bg="#2E2E2E", fg="white",
                                  font=("Verdana", 11))
        self.status_label.pack(pady=5)

    def create_result_area(self):
        # Dividir frame_right em plot na parte superior e texto na parte inferior
        self.frame_plot = Frame(self.frame_right, bg="#2E2E2E", height=400)
        self.frame_plot.pack(fill=BOTH, expand=True)

        self.frame_text = Frame(self.frame_right, bg="#2E2E2E", height=200)
        self.frame_text.pack(fill=BOTH, expand=False)

        # Configurar figura matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor='#2E2E2E')
        self.ax.set_title('Evolução do Fitness (por bloco de migração)', color='white')
        self.ax.set_xlabel('Blocos de Migração', color='white')
        self.ax.set_ylabel('Melhor Fitness', color='white')
        self.ax.tick_params(colors='white')
        self.line1, = self.ax.plot([], [], label='Ilha 1', color='cyan', marker='o')
        self.line2, = self.ax.plot([], [], label='Ilha 2', color='magenta', marker='o')
        self.ax.legend(facecolor='#2E2E2E', labelcolor='white', edgecolor='white')
        self.fig.patch.set_facecolor('#2E2E2E')

        # Canvas Tkinter para embutir matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.canvas.draw()

        # Área de texto para resultados finais
        Label(self.frame_text, text="Resultados Finais:", bg="#2E2E2E", fg="white",
              font=("Verdana", 14, "bold")).pack(anchor="nw", pady=(5,0))
        self.text_area = Text(self.frame_text, bg="#1E1E1E", fg="white",
                              font=("Consolas", 11), wrap="word", height=6)
        scrollbar = Scrollbar(self.frame_text, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.text_area.pack(fill=BOTH, expand=True, padx=5, pady=5)

    def on_run(self):
        """
        Ao clicar em Run:
         1) Desabilita botão e limpa áreas
         2) Lê todos os parâmetros
         3) Instancia duas ilhas com os mesmos parâmetros
         4) Inicializa populações e avaliação inicial para geração 0
         5) Inicia thread que executa toda a evolução com callback na migração
        """
        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Executando...")
        self.text_area.delete(1.0, END)

        # Limpar gráfico de blocos anteriores
        self.ax.clear()
        self.ax.set_title('Evolução do Fitness (por bloco de migração)', color='white')
        self.ax.set_xlabel('Blocos de Migração', color='white')
        self.ax.set_ylabel('Melhor Fitness', color='white')
        self.ax.tick_params(colors='white')
        self.line1, = self.ax.plot([], [], label='Ilha 1', color='cyan', marker='o')
        self.line2, = self.ax.plot([], [], label='Ilha 2', color='magenta', marker='o')
        self.ax.legend(facecolor='#2E2E2E', labelcolor='white', edgecolor='white')
        self.fig.patch.set_facecolor('#2E2E2E')
        self.canvas.draw()

        # Ler parâmetros
        try:
            population_size = int(self.entries['pop_size'].get())
            crossover_rate = float(self.entries['cx_rate'].get())
            mutation_rate = float(self.entries['mu_rate'].get())
            generations = int(self.entries['gens'].get())
            elitism_size = int(self.entries['elitism'].get())
            tournament_size = int(self.entries['tourn'].get())
            use_tournament = self.tournament_var.get()
            self.migration_interval = int(self.entry_mig.get())
            path = self.entry_path.get().strip()
        except ValueError:
            self.status_label.config(text="Status: Erro nos parâmetros")
            self.run_button.config(state=tk.NORMAL)
            return

        # Carregar dados de distâncias e cidades
        try:
            df_distances = load_location_data(path)
            cities = get_route_points(df_distances)
        except Exception as e:
            self.status_label.config(text=f"Status: Erro ao carregar dados ({e})")
            self.run_button.config(state=tk.NORMAL)
            return

        # Instanciar as duas ilhas
        params = {
            'distance_df': df_distances,
            'cities': cities,
            'population_size': population_size,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'generations': generations,
            'elitism_size': elitism_size,
            'use_tournament_selection': use_tournament,
            'tournament_size': tournament_size
        }
        island1 = SalesmanProblemGA(**params)
        island2 = SalesmanProblemGA(**params)
        self.islands = [island1, island2]
        self.total_generations = generations

        # Inicializar populações e avaliação inicial (bloco 0)
        for isl in self.islands:
            isl._initialize_population()
            isl._evaluate_population()  # gera history_best_fitness[0]

        # Iniciar thread de evolução com callback na migração
        self.thread = threading.Thread(target=self.background_evolution, daemon=True)
        self.thread.start()

    def background_evolution(self):
        """
        Executa toda a evolução em thread separada, chamando callback na migração.
        """
        def migration_callback(block, best_fitnesses, best_route, best_fitness):
            # Agendar atualização de gráfico no mainloop
            self.root.after(0, self.update_plot_blocks)

        # Chama multipopulation_evolution com callback
        best_route, best_fitness = multipopulation_evolution(
            self.islands,
            self.migration_interval,
            callback=migration_callback
        )
        self.best_route_final = best_route
        self.best_fitness_final = best_fitness
        # Ao concluir, exibir resultados finais no mainloop
        self.root.after(0, self.show_final_results)

    def update_plot_blocks(self):
        """
        Atualiza o gráfico com os pontos de fitness de cada bloco de migração.
        """
        if not self.islands:
            return
        # Cada ilha possui history_best_fitness por geração, mas queremos usar valor do fim
        # de cada bloco de migração. O tamanho do history é geracoes acumuladas.
        # Para simplificar, vamos extrair fitness nos índices múltiplos de migration_interval - 1.
        hi1 = self.islands[0].history_best_fitness
        hi2 = self.islands[1].history_best_fitness
        blocks = []
        vals1 = []
        vals2 = []
        mi = self.migration_interval
        # O bloco 0 corresponde à avaliação inicial, já em index 0
        n_blocks = len(hi1) // mi
        for b in range(n_blocks):
            idx = b * mi + (mi - 1)
            if idx < len(hi1):
                blocks.append(b + 1)
                vals1.append(hi1[idx])
                vals2.append(hi2[idx])
        # Desenhar
        self.ax.clear()
        self.ax.set_title('Evolução do Fitness (por bloco de migração)', color='white')
        self.ax.set_xlabel('Blocos de Migração', color='white')
        self.ax.set_ylabel('Melhor Fitness', color='white')
        self.ax.tick_params(colors='white')
        if vals1:
            self.ax.plot(blocks, vals1, label='Ilha 1', color='cyan', marker='o')
        if vals2:
            self.ax.plot(blocks, vals2, label='Ilha 2', color='magenta', marker='o')
        self.ax.legend(facecolor='#2E2E2E', labelcolor='white', edgecolor='white')
        self.fig.patch.set_facecolor('#2E2E2E')
        self.canvas.draw()

    def show_final_results(self):
        """
        Exibe no text_area a melhor rota geral, distância e fitness, e reabilita botão Run.
        """
        self.running = False
        fazenda = "Fazenda em Delta - MG"
        if self.best_route_final is not None:
            full_route = [fazenda] + self.best_route_final + [fazenda]
            best_distance = 300.0 - self.best_fitness_final
            self.text_area.insert(END, "Melhor rota geral encontrada:\n")
            self.text_area.insert(END, " -> ".join(full_route) + "\n")
            self.text_area.insert(END, f"Distância total: {best_distance:.2f} km\n")
            self.text_area.insert(END, f"Fitness global: {self.best_fitness_final:.2f}\n")
        try:
            coord_dict = load_coordinates_from_csv('src/data/Planilha para caixeiro viajante mercados Uberaba.csv')
            maps_urls = generate_google_maps_url(full_route, coord_dict)
            if isinstance(maps_urls, list):
                for idx, url in enumerate(maps_urls, 1):
                    self.text_area.insert(END, f"\nLink para rota no Google Maps (parte {idx}):\n{url}\n")
            else:
                self.text_area.insert(END, f"\nLink para rota no Google Maps:\n{maps_urls}\n")
        except Exception as e:
            self.text_area.insert(END, f"\nErro ao gerar link do Google Maps: {e}\n")
        self.status_label.config(text="Status: Concluído")
        self.run_button.config(state=tk.NORMAL)

    def on_close(self):
        """
        Fecha a janela.
        """
        self.running = False
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = InterfaceGA(root)
    root.mainloop()
