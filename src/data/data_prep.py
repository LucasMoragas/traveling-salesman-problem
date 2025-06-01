import pandas as pd

def load_location_data(excel_path='src/data/locations.xlsx'):
    """
    Reads an Excel file and returns a DataFrame with the data.

    Parameters:
    excel_path (str): Path to the Excel file.

    Returns:
    pd.DataFrame: DataFrame containing the Excel data.
    """
    try:
        df = pd.read_excel(excel_path, header=0)
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

def dataframe_to_dict_list(df):
    """
    Converts a pandas DataFrame into a list of dictionaries,
    removing any key (except 'Cidades') where the value is 0.0.

    Parameters:
    df (pd.DataFrame): DataFrame to convert.

    Returns:
    list: A list of dictionaries representing the cleaned rows of the DataFrame.
    """
    try:
        dict_list = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            cleaned_dict = {k: v for k, v in row_dict.items() if k == 'Cidades' or v != 0.0}
            dict_list.append(cleaned_dict)
        return dict_list
    except Exception as e:
        print(f"Error converting DataFrame to cleaned list of dicts: {e}")
        return []
      
def get_city_names(df):
    """
    Extracts a list of city names from the 'Cidades' column of the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'Cidades' column.

    Returns:
    list: A list of city names.
    """
    try:
        if 'Cidades' not in df.columns:
            raise ValueError("Column 'Cidades' not found in DataFrame.")
        return df['Cidades'].dropna().tolist()
    except Exception as e:
        print(f"Error extracting city names: {e}")
        return []

def get_route_points(df):
    """
    Extracts a list of city names from the 'Cidades' column of the DataFrame,
    excluding 'Fazenda em Delta - MG'.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'Cidades' column.

    Returns:
    list: A list of city names excluding 'Fazenda em Delta - MG'.
    """
    try:
        cities = get_city_names(df)
        filtered_cities = [city for city in cities if city != 'Fazenda em Delta - MG']
        return filtered_cities
    except Exception as e:
        print(f"Error extracting route points: {e}")
        return []
    
def generate_random_combinations(route_points, n=5):
    """
    Generates n shuffled lists of route points.

    Parameters:
    route_points (list): List of route points.
    n (int): Number of shuffles to generate.

    Returns:
    list: A list containing n shuffled lists of route points.
    """
    import random
    combinations = []
    for _ in range(n):
        shuffled = route_points[:]  # copy of the original list
        random.shuffle(shuffled)
        combinations.append(shuffled)
    return combinations

def calculate_distance(df, route):
    """
    Calculates the total distance of a given route based on the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing distances between cities.
    route (list): List of city names representing the route.

    Returns:
    float: Total distance of the route.
    """
    # we will initialize the total distance with the distance from Fazenda em Delta - MG to the first city in the route
    total_distance = 0.0
    try:
        # Start from Fazenda em Delta - MG to the first city
        start_city = 'Fazenda em Delta - MG'
        if not route:
            return 0.0
        # Distance from Fazenda em Delta - MG to the first city
        start_row = df[df['Cidades'] == start_city]
        if start_row.empty or route[0] not in df.columns:
            raise ValueError(f"Cannot find distance from {start_city} to {route[0]}")
        total_distance += float(start_row[route[0]].values[0])

        # Distances between consecutive cities in the route
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            row = df[df['Cidades'] == from_city]
            if row.empty or to_city not in df.columns:
                raise ValueError(f"Cannot find distance from {from_city} to {to_city}")
            total_distance += float(row[to_city].values[0])

        # Finally, add the distance from the last city in the route back to Fazenda em Delta - MG
        end_row = df[df['Cidades'] == route[-1]]
        if end_row.empty or start_city not in df.columns:
            raise ValueError(f"Cannot find distance from {route[-1]} to {start_city}")
        total_distance += float(end_row[start_city].values[0])

        return round(total_distance, 2)
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None
    

if __name__ == "__main__":
    # Example usage
    df = load_location_data()
    route_points = get_route_points(df)
    print(route_points)
    print(len(route_points))
    random_combinations = generate_random_combinations(route_points, n=1000)
    distances = []
    for i, combo in enumerate(random_combinations):
        print(f"Combination: {i + 1}")
        distance = calculate_distance(df, combo)
        print(f"Distance: {distance} km")
        distances.append(distance)
    
    print(f"Max distance: {max(distances)} km")
    print(f"Min distance: {min(distances)} km")