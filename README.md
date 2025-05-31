# Flowshop_AlgoritmoGenetico
Algoritmo Genetico para problemas de programacion Flowshop
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd

# Si no tienes scheptk, puedes definir una función simple:
def sorted_index_asc(lst):
    return sorted(range(len(lst)), key=lst.__getitem__)

# --- Código de GA.py (las funciones insertion y GA) ---
def insertion(sequence):
    vecino = deepcopy(sequence)
    posicion = random.randint(0, len(sequence)-1)

    insertar= vecino.pop(posicion)
    posicion2 = random.randint(0, len(sequence)-1)

    while posicion2 == posicion:
        posicion2 = random.randint(0, len(sequence)-1)

    vecino.insert(posicion2, insertar)

    return vecino

def GA (instancia, ind_special, Objetivo, popsize, iteraciones):
    population=[]
    fitnesspop = []

    bestind = deepcopy(ind_special)
    # Modified to unpack the tuple returned by the objective function
    bestfitness, _ = eval("instancia." + str(Objetivo) + "(bestind)")

    population.append(bestind)
    fitnesspop.append(bestfitness)

    for j in range(popsize-1):
        newindiv = instancia.random_solution()
        population.append(newindiv)
        # Modified to unpack the tuple
        newfitness, _ = eval("instancia." + str(Objetivo) + "(newindiv)")
        fitnesspop.append(newfitness)
        if newfitness < bestfitness:
            bestind = deepcopy(newindiv)
            bestfitness = newfitness

    while iteraciones > 0:
        mejor = sorted_index_asc(fitnesspop)[0]
        padre1= population[mejor]

        aleatorio = random.randint(0, popsize-1)
        while aleatorio == mejor:
            aleatorio = random.randint(0, popsize-1)

        padre2= population[aleatorio]

        hijo=padre1[:(instancia.jobs//2)]
        for i in padre2:
            if i not in hijo:
                hijo.append(i)

        hijo_mutado = insertion(hijo)

        # Modified to unpack the tuple
        fitnesshijomutado, _ = eval("instancia." + str(Objetivo) + "(hijo_mutado)")

        peor = sorted_index_asc(fitnesspop)[-1]
        if fitnesshijomutado < fitnesspop[peor]:
            population.pop(peor)
            population.append(hijo_mutado)
            fitnesspop.pop(peor)
            fitnesspop.append(fitnesshijomutado)

        if fitnesshijomutado < bestfitness:
            bestind = deepcopy(hijo_mutado)
            bestfitness = fitnesshijomutado

        iteraciones = iteraciones - 1

    # Recalculate schedule details for the best individual before returning
    _, best_schedule_details = eval("instancia." + str(Objetivo) + "(bestind)")

    return bestind, bestfitness, best_schedule_details


# --- Clase para la instancia de Flowshop con SDST, Release Dates y Due Dates ---
class FlowshopWithAllFeatures:
    def __init__(self, processing_times, setup_times, due_dates, weights, release_dates):
        """
        Inicializa la instancia del problema Flowshop con SDST, fechas de lanzamiento y fechas de vencimiento.

        :param processing_times: Una lista de listas. processing_times[job_idx][machine_idx] es el tiempo de procesamiento.
                                 (Indices de trabajo 0-5, máquina 0-3)
        :param setup_times: Una lista 3D. setup_times[prev_job_idx_plus_1][current_job_idx_plus_1][machine_idx]
                            es el tiempo de setup al cambiar de prev_job_id a current_job_id en machine_id.
                            prev_job_idx_plus_1 y current_job_idx_plus_1 usan 1-based indexing como en el PDF.
                            (Indices de trabajo 0-6 para prev/current job, máquina 0-3)
        :param due_dates: Diccionario {job_idx: due_date}. (Indices de trabajo 0-5)
        :param weights: Diccionario {job_idx: weight}. (Indices de trabajo 0-5)
        :param release_dates: Diccionario {job_idx: release_date}. (Indices de trabajo 0-5)
        """
        self.processing_times = processing_times # Trabajos 0-5, Máquinas 0-3
        self.setup_times = setup_times # Trabajos 0-6 (0=idle), Máquinas 0-3 (M1-M4)
        self.due_dates = due_dates #
        self.weights = weights #
        self.release_dates = release_dates #

        self.num_jobs = len(processing_times) # 6 trabajos (0-5)
        self.num_machines = len(processing_times[0]) # 4 máquinas (0-3)
        self.jobs = self.num_jobs # Atributo 'jobs' requerido por el GA

        # Mapeo de IDs de trabajo del PDF (1-6) a índices de lista (0-5)
        # y para la matriz de setup que usa 1-6 para trabajos y 0 para "idle"
        self.pdf_job_to_list_idx = {i+1: i for i in range(self.num_jobs)}
        self.list_idx_to_pdf_job = {i: i+1 for i in range(self.num_jobs)}

        # Para la matriz de setup: el índice 0 en setup_times es el estado "idle" o "ningún trabajo previo"
        self.IDLE_JOB_IDX_SETUP_MATRIX = 0 # Corresponde al índice 0 en las matrices Sijk del PDF

    def random_solution(self):
        """
        Genera una permutación aleatoria de los trabajos (índices 0 a num_jobs-1) como una solución.
        """
        solution = list(range(self.num_jobs))
        random.shuffle(solution)
        return solution

    def calculate_weighted_earliness_tardiness(self, sequence):
        """
        Calcula la función objetivo: Suma de la anticipación ponderada y la tardanza ponderada.

        :param sequence: Una lista que representa el orden de los trabajos (índices 0 a num_jobs-1).
        :return: El valor de la función objetivo y una lista de diccionarios con los detalles de la programación.
        """
        if not sequence:
            return 0, []

        # completion_times[job_id][machine_id]
        completion_times = [[0] * self.num_machines for _ in range(self.num_jobs)]

        # Tiempo de disponibilidad de la máquina (cuando la máquina termina el trabajo anterior)
        machine_available_time = [0] * self.num_machines

        # Último trabajo (ID de PDF 1-6) completado en cada máquina para calcular SDST
        last_job_pdf_id_on_machine = [self.IDLE_JOB_IDX_SETUP_MATRIX] * self.num_machines

        # Tiempos de finalización de cada trabajo (para calcular earliness/tardiness)
        job_finish_times = [0] * self.num_jobs

        # Store schedule details for Gantt chart
        schedule_details = []

        for i in range(self.num_jobs):
            current_job_list_idx = sequence[i] # Índice del trabajo en la secuencia (0-5)
            current_job_pdf_id = self.list_idx_to_pdf_job[current_job_list_idx] # ID del trabajo en PDF (1-6)

            job_current_release_time = self.release_dates[current_job_list_idx]

            for j in range(self.num_machines): # Iterar sobre las máquinas de procesamiento (0 a num_machines-1, que es 0-3)
                # Tiempo de finalización del trabajo actual en la máquina anterior
                prev_machine_finish_time = completion_times[current_job_list_idx][j-1] if j > 0 else 0

                # Obtener el tiempo de setup SDST
                prev_job_pdf_id_on_this_machine = last_job_pdf_id_on_machine[j]

                # Acceder a la matriz de setup: setup_times[prev_job_pdf_id][current_job_pdf_id][machine_idx]
                # El machine_idx aquí es 'j' porque estamos en la máquina actual (0-3).
                current_setup_time = self.setup_times[prev_job_pdf_id_on_this_machine][current_job_pdf_id][j]

                # Tiempo en que la máquina está disponible (después de terminar el trabajo anterior en esa máquina)
                machine_ready_time = machine_available_time[j]

                # El trabajo no puede comenzar hasta que:
                # 1. La máquina esté disponible (incluido el setup).
                # 2. El trabajo haya terminado en la máquina anterior (si no es la primera máquina).
                # 3. La fecha de lanzamiento del trabajo haya llegado (solo para la primera máquina).

                if j == 0: # Primera máquina (M1)
                    job_start_time_on_machine = max(machine_ready_time + current_setup_time,
                                                    prev_machine_finish_time, # Será 0 para la primera máquina
                                                    job_current_release_time)
                else: # Máquinas subsiguientes (M2, M3, M4)
                    job_start_time_on_machine = max(machine_ready_time + current_setup_time,
                                                    prev_machine_finish_time)

                job_finish_time_on_machine = job_start_time_on_machine + self.processing_times[current_job_list_idx][j]

                completion_times[current_job_list_idx][j] = job_finish_time_on_machine
                machine_available_time[j] = job_finish_time_on_machine
                last_job_pdf_id_on_machine[j] = current_job_pdf_id # Actualizar el último trabajo en esta máquina

                # Store details for Gantt chart
                schedule_details.append({
                    "Trabajo": current_job_list_idx + 1, # Convert to 1-based for display
                    "Máquina": f"M{j+1}", # Using M1, M2, etc. for machine names
                    "Inicio": job_start_time_on_machine,
                    "Fin": job_finish_time_on_machine,
                    "Etapa": f"Machine {j+1}" # Simple stage name for now
                })

            # El tiempo de finalización del trabajo es el tiempo de finalización en la última máquina de procesamiento
            job_finish_times[current_job_list_idx] = completion_times[current_job_list_idx][self.num_machines - 1]

        # Calcular la función objetivo (anticipación ponderada y tardanza ponderada)
        total_objective_value = 0
        for job_list_idx in range(self.num_jobs):
            C_j = job_finish_times[job_list_idx]
            d_j = self.due_dates[job_list_idx]
            w_j = self.weights[job_list_idx]

            earliness = max(0, d_j - C_j)
            tardiness = max(0, C_j - d_j)

            total_objective_value += w_j * earliness + w_j * tardiness

        return total_objective_value, schedule_details

def graficar_programacion_maquinas(df):
    df = df[df["Máquina"].notnull()]

    # Map 'Etapa' based on 'Máquina' (M1, M2, M3, M4)
    # The current problem uses M1-M4 as machines, not specific stages like Mezcladora.
    # So, we'll map directly from machine name to a numeric order.
    # Assuming M1, M2, M3, M4 are the stages in that order.
    stage_order = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}

    # Mapear cada máquina a su etapa (for sorting purposes)
    machine_stage = {}
    for _, row in df.iterrows():
        m = row["Máquina"]
        if m not in machine_stage:
            machine_stage[m] = m # Here, machine name is its own "stage" for ordering

    # Order machines based on the defined stage_order, then alphabetically
    maquinas = list(machine_stage.keys())
    maquinas.sort(key=lambda m: (stage_order.get(machine_stage[m], 999), m), reverse=True)

    trabajos = sorted(df["Trabajo"].unique())
    cmap = plt.cm.get_cmap('Set1', len(trabajos))
    colores = {trabajo: cmap(i) for i, trabajo in enumerate(trabajos)}
    posiciones = {maquina: i for i, maquina in enumerate(maquinas)}

    fig, ax = plt.subplots(figsize=(12, len(maquinas) * 0.8))
    bar_height = 0.4

    for _, row in df.iterrows():
        inicio, fin = row["Inicio"], row["Fin"]
        duracion = fin - inicio
        y = posiciones[row["Máquina"]]
        ax.broken_barh([(inicio, duracion)], (y - bar_height/2, bar_height),
                       facecolors=colores[row["Trabajo"]],
                       edgecolors='black', linewidth=1)
        ax.text(inicio + duracion/2, y, str(row["Trabajo"]),
                ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    ax.set_xlabel("Tiempo") # Unit is minutes based on the problem data
    ax.set_ylabel("Máquinas")
    ax.set_yticks(list(posiciones.values()))
    ax.set_yticklabels(list(posiciones.keys()))
    ax.set_title("Programación de Flowshop con SDST, Release Dates y Due Dates")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# --- Datos de los PDFs ---

# 1. Tiempos de Proceso (Trabajos 1-6, Máquinas M1-M4)
# Mapeado a índices 0-5 para trabajos y 0-3 para máquinas
processing_times_data = [
    [625, 1250, 625, 250],  # Job 1 (idx 0)
    [750, 1500, 1125, 250], # Job 2 (idx 1)
    [750, 1625, 1375, 375], # Job 3 (idx 2)
    [1000, 1750, 1625, 375],# Job 4 (idx 3)
    [1000, 2625, 1625, 375],# Job 5 (idx 4)
    [1125, 2875, 1500, 500] # Job 6 (idx 5)
]

# 2. Tiempos de Setup Sijk (Trabajos 1-6, Máquinas M1-M4)
# setup_times[prev_job_pdf_id][current_job_pdf_id][machine_idx]
# prev_job_pdf_id y current_job_pdf_id van de 0 (idle) a 6.
# machine_idx va de 0 a 3 (para M1-M4).

# Datos de Setup de M1 a M4
M1_setups = [
    [0, 6, 6, 15, 15, 15],
    [6, 0, 6, 15, 15, 15],
    [6, 6, 0, 15, 15, 15],
    [15, 15, 15, 0, 5, 6],
    [15, 15, 15, 5, 0, 6],
    [15, 15, 15, 6, 6, 0]
]

M2_setups = [
    [0, 5, 5, 12, 12, 12],
    [5, 0, 5, 12, 12, 12],
    [5, 5, 0, 12, 12, 12],
    [12, 12, 12, 0, 4, 5],
    [12, 12, 12, 4, 0, 5],
    [12, 12, 12, 5, 5, 0]
]

M3_setups = [
    [0, 7, 7, 14, 14, 14],
    [7, 0, 7, 14, 14, 14],
    [7, 7, 0, 14, 14, 14],
    [14, 14, 14, 0, 6, 7],
    [14, 14, 14, 6, 0, 7],
    [14, 14, 14, 7, 7, 0]
]

M4_setups = [
    [0, 4, 4, 10, 10, 10],
    [4, 0, 4, 10, 10, 10],
    [4, 4, 0, 10, 10, 10],
    [10, 10, 10, 0, 3, 4],
    [10, 10, 10, 3, 0, 4],
    [10, 10, 10, 4, 4, 0]
]


# Crear la matriz 3D setup_times[prev_job_pdf_id][current_job_pdf_id][machine_idx]
# Los IDs de trabajo en el PDF son 1-6. El índice 0 se usará para 'idle'
# Dimensiones: (num_jobs + 1) x (num_jobs + 1) x num_total_setup_machines (4 en este caso)
num_jobs_pdf_ids = 6 # Jobs 1-6
num_total_setup_machines = 4 # M1-M4 para setups

# Inicializar con ceros, tamaño (6+1)x(6+1)x4
setup_times_data = [[[0 for _ in range(num_total_setup_machines)]
                     for _ in range(num_jobs_pdf_ids + 1)]
                    for _ in range(num_jobs_pdf_ids + 1)]

# Rellenar la matriz de setup. Los índices de la lista M_setups son 0-5 para prev/current job 1-6.
# Los índices de setup_times_data son 1-6 para prev/current job.
# setup_times_data[prev_job_pdf_id][current_job_pdf_id][machine_idx]
for prev_j_list_idx in range(num_jobs_pdf_ids):
    for curr_j_list_idx in range(num_jobs_pdf_ids):
        prev_j_pdf_id = prev_j_list_idx + 1
        curr_j_pdf_id = curr_j_list_idx + 1

        setup_times_data[prev_j_pdf_id][curr_j_pdf_id][0] = M1_setups[prev_j_list_idx][curr_j_list_idx] # M1
        setup_times_data[prev_j_pdf_id][curr_j_pdf_id][1] = M2_setups[prev_j_list_idx][curr_j_list_idx] # M2
        setup_times_data[prev_j_pdf_id][curr_j_pdf_id][2] = M3_setups[prev_j_list_idx][curr_j_list_idx] # M3
        setup_times_data[prev_j_pdf_id][curr_j_pdf_id][3] = M4_setups[prev_j_list_idx][curr_j_list_idx] # M4


# 3. Fechas de Vencimiento y Pesos (Trabajos 1-6)
# Mapeado a índices 0-5 para trabajos
due_dates_data = {
    0: 1440, # Job 1
    1: 2640, # Job 2
    2: 2280, # Job 3
    3: 3360, # Job 4
    4: 3120, # Job 5
    5: 3840  # Job 6
}

weights_data = {
    0: 3, # Job 1
    1: 4, # Job 2
    2: 5, # Job 3
    3: 2, # Job 4
    4: 5, # Job 5
    5: 1  # Job 6
}

# 4. Fechas de Lanzamiento (Trabajos 1-6)
# Mapeado a índices 0-5 para trabajos
release_dates_data = {
    0: 0,    # Job 1
    1: 2000, # Job 2
    2: 480,  # Job 3
    3: 1920, # Job 4
    4: 1080, # Job 5
    5: 2280  # Job 6
}

# --- Ejemplo de uso con todos los datos ---
if __name__ == "__main__":
    # Crear una instancia del problema Flowshop con todas las características
    flowshop_problem_all_features = FlowshopWithAllFeatures(
        processing_times=processing_times_data,
        setup_times=setup_times_data,
        due_dates=due_dates_data,
        weights=weights_data,
        release_dates=release_dates_data
    )

    # Definir un individuo especial. Podría ser una secuencia inicial aleatoria.
    initial_special_individual = list(range(flowshop_problem_all_features.num_jobs))
    random.shuffle(initial_special_individual)

    # Parámetros para el Algoritmo Genético
    population_size = 100
    num_iterations = 1000
    objective_function_name = "calculate_weighted_earliness_tardiness"

    print("--- Ejecutando AG con Flowshop, SDST, Release Dates y Weighted Earliness/Tardiness ---")
    best_sequence_all_features, min_objective_value, best_schedule_details = GA(
        instancia=flowshop_problem_all_features,
        ind_special=initial_special_individual,
        Objetivo=objective_function_name,
        popsize=population_size,
        iteraciones=num_iterations
    )

    print(f"\nMejor secuencia encontrada: {best_sequence_all_features}")
    print(f"Valor mínimo de la función objetivo (Anticipación/Tardanza ponderada): {min_objective_value}")

    # Convert best_schedule_details to DataFrame and plot
    df_schedule = pd.DataFrame(best_schedule_details)
    graficar_programacion_maquinas(df_schedule)

    # Opcional: Probar una secuencia específica y ver su FO
    test_sequence = [0, 1, 2, 3, 4, 5] # Ejemplo: Orden natural de los trabajos
    test_objective, test_schedule_details = flowshop_problem_all_features.calculate_weighted_earliness_tardiness(test_sequence)
    print(f"\nValor de la función objetivo para secuencia {test_sequence}: {test_objective}")
    # df_test_schedule = pd.DataFrame(test_schedule_details)
    # graficar_programacion_maquinas(df_test_schedule) # Uncomment to plot this test sequence

    test_sequence_2 = [5, 4, 3, 2, 1, 0] # Ejemplo: Orden inverso
    test_objective_2, test_schedule_details_2 = flowshop_problem_all_features.calculate_weighted_earliness_tardiness(test_sequence_2)
    print(f"Valor de la función objetivo para secuencia {test_sequence_2}: {test_objective_2}")
    # df_test_schedule_2 = pd.DataFrame(test_schedule_details_2)
    # graficar_programacion_maquinas(df_test_schedule_2) # Uncomment to plot this test sequence
