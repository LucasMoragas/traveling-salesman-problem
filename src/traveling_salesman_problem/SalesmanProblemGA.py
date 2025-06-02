import random
import copy
from concurrent.futures import ThreadPoolExecutor
from src.data.data_prep import calculate_distance
from src.data.data_prep import load_location_data, get_route_points, load_cords_data


class SalesmanProblemGA:
    """
    Implementação de um Algoritmo Genético para o Problema do Caixeiro Viajante (TSP),
    com crossover PMX, mutação por troca, seleção por torneio ou roleta e elitismo.
    A função de aptidão é definida como: fitness = 300 - distância_total (quanto menor a distância, maior o fitness).
    """

    def __init__(
        self,
        distance_df,
        cities,
        population_size=20,
        crossover_rate=0.8,
        mutation_rate=0.02,
        generations=50,
        elitism_size=2,
        use_tournament_selection=True,
        tournament_size=5,
    ):
        """
        Args:
            distance_df (pd.DataFrame): DataFrame contendo a matriz de distâncias entre cidades.
            cities (list[str]): Lista de nomes de cidades a serem visitadas (excluindo a cidade-base).
            population_size (int): Número de indivíduos na população.
            crossover_rate (float): Probabilidade de aplicar crossover PMX entre dois pais.
            mutation_rate (float): Probabilidade de mutação para cada gene.
            generations (int): Número de gerações a evoluir.
            elitism_size (int): Quantidade de indivíduos melhores preservados em cada geração.
            use_tournament_selection (bool): Se True, usa seleção por torneio; senão, seleção por roleta.
            tournament_size (int): Tamanho do torneio para seleção.
        """
        self.distance_df = distance_df
        self.cities = list(cities)
        self.num_cities = len(self.cities)

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.elitism_size = elitism_size
        self.use_tournament = use_tournament_selection
        self.tournament_size = tournament_size

        self.history_best_fitness = []
        self.history_best_route = []

        self.population = []

    def _initialize_population(self):
      """
      Inicializa a população com cromossomos aleatórios (rotas aleatórias).
      Cada cromossomo é uma permutação da lista de cidades.
      """
      pop = []
      for _ in range(self.population_size):
        chromosome = self.cities[:]  # Cria uma cópia da lista de cidades
        random.shuffle(chromosome)   # Embaralha para gerar uma rota aleatória
        pop.append(chromosome)       # Adiciona à população
      self.population = pop            # Define a população inicial

    def _route_distance(self, route):
      # Calcula a distância total de uma rota usando a matriz de distâncias
      return calculate_distance(self.distance_df, route)

    def _fitness(self, route):
      # Calcula o fitness de uma rota: quanto menor a distância, maior o fitness
      dist = self._route_distance(route)
      if dist is None:
        return -1e6  # Penalidade para rotas inválidas
      return 300.0 - dist  # Fitness inversamente proporcional à distância

    def _evaluate_population(self):
      # Avalia toda a população, retornando listas de fitness, distâncias,
      # o melhor indivíduo e seu fitness
      fitnesses = []
      distances = []
      for chromo in self.population:
        dist = self._route_distance(chromo)
        fit = self._fitness(chromo)
        distances.append(dist)
        fitnesses.append(fit)
      best_idx = max(range(len(self.population)), key=lambda i: fitnesses[i])
      best_route = self.population[best_idx][:]
      best_fit = fitnesses[best_idx]
      return fitnesses, distances, best_route, best_fit

    def _tournament_selection(self, fitnesses):
      # Seleção por torneio: escolhe aleatoriamente 'tournament_size' indivíduos
      # e retorna o melhor entre eles
      selected_indices = random.sample(range(self.population_size), self.tournament_size)
      best = selected_indices[0]
      for idx in selected_indices[1:]:
        if fitnesses[idx] > fitnesses[best]:
          best = idx
      return self.population[best][:]

    def _roulette_wheel_selection(self, fitnesses):
      # Seleção por roleta: probabilidade proporcional ao fitness (ajustado para evitar valores negativos)
      min_fit = min(fitnesses)
      shift = abs(min_fit) + 1e-6 if min_fit < 0 else 0.0  # Garante que todos os fitness sejam positivos
      adjusted = [f + shift for f in fitnesses]
      total = sum(adjusted)
      if total == 0:
        # Se todos os fitness são zero, seleciona aleatoriamente
        idx = random.randrange(self.population_size)
        return self.population[idx][:]
      pick = random.uniform(0, total)
      current = 0.0
      for idx, af in enumerate(adjusted):
        current += af
        if current >= pick:
          return self.population[idx][:]
      return self.population[-1][:]  # Fallback

    def _select_parent(self, fitnesses):
      # Decide qual método de seleção usar (torneio ou roleta)
      if self.use_tournament:
        return self._tournament_selection(fitnesses)
      else:
        return self._roulette_wheel_selection(fitnesses)

    def _pmx_crossover(self, parent1, parent2):
      # Crossover PMX (Partially Mapped Crossover) para TSP
      size = self.num_cities
      pt1 = random.randrange(0, size)
      pt2 = random.randrange(0, size)
      if pt1 > pt2:
        pt1, pt2 = pt2, pt1
      child1 = [None] * size
      child2 = [None] * size
      # Copia o segmento do segundo pai para o primeiro filho e vice-versa
      for i in range(pt1, pt2 + 1):
        child1[i] = parent2[i]
        child2[i] = parent1[i]
      # Preenche o restante dos genes, mantendo a ordem e evitando duplicatas
      for i in list(range(0, pt1)) + list(range(pt2 + 1, size)):
        gene = parent1[i]
        while gene in parent2[pt1:pt2 + 1]:
          idx = parent2.index(gene)
          gene = parent1[idx]
        child1[i] = gene
      for i in list(range(0, pt1)) + list(range(pt2 + 1, size)):
        gene = parent2[i]
        while gene in parent1[pt1:pt2 + 1]:
          idx = parent1.index(gene)
          gene = parent2[idx]
        child2[i] = gene
      return child1, child2

    def _crossover(self, parent1, parent2):
      # Aplica crossover PMX com probabilidade crossover_rate
      if random.random() < self.crossover_rate:
        return self._pmx_crossover(parent1, parent2)
      else:
        # Sem crossover: filhos são cópias dos pais
        return parent1[:], parent2[:]

    def _mutate(self, chromosome):
      # Mutação por troca: para cada gene, com probabilidade mutation_rate, troca com outro gene aleatório
      size = self.num_cities
      for i in range(size):
        if random.random() < self.mutation_rate:
          j = random.randrange(0, size)
          chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    def _create_next_generation(self, fitnesses):
      # Cria a próxima geração usando elitismo, seleção, crossover e mutação
      # 1. Elitismo: mantém os melhores indivíduos
      sorted_indices = sorted(range(len(self.population)), key=lambda i: fitnesses[i], reverse=True)
      next_population = []
      for i in range(self.elitism_size):
        elite_idx = sorted_indices[i]
        next_population.append(self.population[elite_idx][:])
      # 2. Preenche o restante da população
      while len(next_population) < self.population_size:
        # Seleciona dois pais
        parent1 = self._select_parent(fitnesses)
        parent2 = self._select_parent(fitnesses)
        # Aplica crossover
        child1, child2 = self._crossover(parent1, parent2)
        # Aplica mutação
        self._mutate(child1)
        self._mutate(child2)
        # Adiciona filhos à próxima geração
        next_population.append(child1)
        if len(next_population) < self.population_size:
          next_population.append(child2)
      self.population = next_population

    def evolve(self, num_generations=10):
        """
        Evolui a população por um número específico de gerações.
        Retorna o melhor indivíduo e seu fitness ao final do bloco.
        """
        best_route = None
        best_fitness = -float("inf")
        for _ in range(num_generations):
            fitnesses, distances, gen_best_route, gen_best_fit = self._evaluate_population()
            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_route = gen_best_route[:]
            self._create_next_generation(fitnesses)
        return best_route, best_fitness

    def run(self):
      """
      Executa o algoritmo genético do início ao fim, inicializando a população e evoluindo por todas as gerações.
      Retorna o melhor indivíduo encontrado e seu fitness.
      """
      self._initialize_population()  # Inicializa a população com rotas aleatórias
      best_overall = None
      best_fitness_overall = -float("inf")
      for gen in range(1, self.generations + 1):
        # Avalia a população atual
        fitnesses, distances, gen_best_route, gen_best_fit = self._evaluate_population()
        # Atualiza o melhor indivíduo global se necessário
        if gen_best_fit > best_fitness_overall:
          best_fitness_overall = gen_best_fit
          best_overall = gen_best_route[:]
        # Gera a próxima geração, exceto na última iteração
        if gen < self.generations:
          self._create_next_generation(fitnesses)
      return best_overall, best_fitness_overall


def multipopulation_evolution(islands, migration_interval=10):
    """
    Recebe uma lista de instâncias de SalesmanProblemGA (ilhas) e executa a evolução paralela.
    A cada 'migration_interval' gerações, o melhor indivíduo de cada ilha é migrado para a próxima ilha.
    Retorna o melhor indivíduo geral e seu fitness.
    """
    num_islands = len(islands)
    # Inicializa populações de todas as ilhas
    for island in islands:
        island._initialize_population()

    total_blocks = islands[0].generations // migration_interval
    best_global_route = None
    best_global_fitness = -float("inf")

    for block in range(total_blocks):
        # Evoluir cada ilha por 'migration_interval' gerações em paralelo
        with ThreadPoolExecutor(max_workers=num_islands) as executor:
            futures = [executor.submit(island.evolve, migration_interval) for island in islands]
            results = [f.result() for f in futures]

        # Extrair melhores de cada ilha e atualizar o global
        best_routes = []
        best_fitnesses = []
        for route, fit in results:
            best_routes.append(route)
            best_fitnesses.append(fit)
            if fit > best_global_fitness:
                best_global_fitness = fit
                best_global_route = route[:]

        # Migração: substituir pior de cada ilha pelo melhor da ilha anterior
        for i, island in enumerate(islands):
            # Avaliar fitness atual para encontrar o pior
            fitnesses, _, _, _ = island._evaluate_population()
            worst_idx = min(range(len(island.population)), key=lambda idx: fitnesses[idx])
            best_from_prev = best_routes[(i - 1) % num_islands][:]
            island.population[worst_idx] = best_from_prev

        print(f"Após bloco {block + 1}, melhores fitness por ilha: {best_fitnesses}")

    return best_global_route, best_global_fitness

def get_route_points_cords(df, route):
    """
    Obtém as coordenadas de latitude e longitude para os pontos de uma rota.
    
    Args:
        df (pd.DataFrame): DataFrame contendo as coordenadas das cidades.
        route (list): Lista de nomes de cidades representando a rota.

    Returns:
        list: Lista de tuplas com as coordenadas (latitude, longitude) dos pontos da rota.
    """
    coords = []
    for city in route:
        row = df[df['Cidades'] == city]
        if not row.empty:
            coords.append((row['Coordenadas'].values[0]))
    return coords

def generate_google_maps_url(route_coords):
    """
    Gera uma URL do Google Maps para a rota especificada pelas coordenadas.

    Args:
        route_coords (list): Lista de strings com as coordenadas no formato 'lat, lon'.

    Returns:
        str: URL do Google Maps com a rota.
    """
    base_url = "https://www.google.com/maps/dir/?api=1"
    start_end_cords = '-19.978458, -47.807004'  # Coordenadas da Fazenda em Delta - MG inicio e fim da rota
    # Remove espaços das coordenadas e monta a string de waypoints separada por '|'
    waypoints = '|'.join(coord.replace(' ', '') for coord in route_coords)
    # Origem e destino são fixos (Fazenda em Delta - MG)
    url = f"{base_url}&origin={start_end_cords.replace(' ', '')}&destination={start_end_cords.replace(' ', '')}&waypoints={waypoints}&travelmode=driving"
    return url
  

if __name__ == "__main__":
    path = 'src/data/locations.xlsx'
    df_distances = load_location_data(path)
    cities = get_route_points(df_distances)

    # Configurar duas ilhas
    island1 = SalesmanProblemGA(
        distance_df=df_distances,
        cities=cities,
        population_size=20,
        crossover_rate=0.9,
        mutation_rate=0.05,
        generations=200,
        elitism_size=2,
        use_tournament_selection=True,
        tournament_size=3
    )
    island2 = SalesmanProblemGA(
        distance_df=df_distances,
        cities=cities,
        population_size=20,
        crossover_rate=0.9,
        mutation_rate=0.05,
        generations=200,
        elitism_size=2,
        use_tournament_selection=True,
        tournament_size=3
    )

    # Executar evolução multipopulação
    best_route, best_fitness = multipopulation_evolution([island1, island2], migration_interval=20)

    print("Melhor rota geral encontrada:")
    print(" -> ".join(best_route))
    print(f"Fitness global: {best_fitness:.2f}")
    print("Distância total da melhor rota:", calculate_distance(df_distances, best_route))
    
    df_coords = load_cords_data()
    best_route_coords = get_route_points_cords(df_coords, best_route)
    
    print("Coordenadas da melhor rota:")
    print(best_route_coords)
    
    google_maps_url = generate_google_maps_url(best_route_coords)
    print("URL do Google Maps para a rota:")
    print(google_maps_url)
        
