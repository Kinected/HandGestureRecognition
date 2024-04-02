import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Temps d'exécution de {func.__name__}: {end_time - start_time} secondes")
        return result
    return wrapper