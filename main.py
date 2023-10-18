import math
import numpy as np
import scipy

def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r == 0 or h == 0:
        return None
    else:
        surf = 2 * math.pi * r * (r + h)
    return surf

def aritm_arange(start: float, stop: float, step: float):
    if step != 1 and type(step) != int:
        aritmetical = np.arange(start, stop, step)
    else: return None
    return aritmetical

def aritm_linear(start: float, stop: float, num: int):
    if num != 1:
        aritmetical = np.linear(start, stop, num)
    else: return None
    return aritmetical





def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    if n < 0 or type(n) != int:
        return None
    List = []
    if n < 1:
        List = [0]
    else:
        List = [0, 1]
        for i in range(2,n+1):
            List.append(List[i-1] + List[i-2])        
    fib_elem = np.ndarray(List)
    return fib_elem


def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    A = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    det_A = np.linalg.det(A)
    A_T = np.transpose(A)
    if det_A != 0:
        A_M = np.inv(A)
    else: A_M = None
    A_touple = (A_M, A_T, det_A)
    return A_touple

def vectors_matrix():
    M = np.array([[3, 1, -2, 4],[0, 1, 1, 5], [-2, 1, 1, 6], [4, 3, 0, 1]])
    print(M[0,0], M[2,2], M[2,1])
    w1 = M[:, 2]
    w2 = M[1, :]



def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    M = np.zeros((m, n))
    for i in range(m):
        for j in range (n):
            if i > j:
                M[i,j] = i
            else:
                M[i, j] = j
    return M


def numpy_functions():
    v1 = np.array([[1], [3], [13]])
    v2 = np.array([[8], [5], [-2]])
    d1 = np.multiply(4,v1)
    d2 = -v2 + np.array([[2], [2], [2]])
    d3 = np.matmul(v1, np.transpose(v2))
    d4 = np.multiply(v1, v2)

    M = np.array([[1, -7, 3], [-12, 3, 4], [5, 13, -3]])
    d11 = np.multiply(3,M)
    d12 = d1 + np.ones(3,3)
    d13 = np.transpose(M)
    d14 = np.matmul(M, v1)
    d15 = np.matmul(np.transpose(v2), M)