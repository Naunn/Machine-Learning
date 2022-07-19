# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 22:54:43 2022

@author: Bartosz Lewandowski
"""

from random import choice

class Puzzle:
    
    # Pustka := puste pole w zagadce
    # Zmienne poruszania sie "pustki"
    UP, DOWN, LEFT, RIGHT = (1,0), (-1,0), (0,1), (0,-1)    
    
    DIRECTIONS = [UP, DOWN, LEFT, RIGHT]
    
    def __init__(self, board_size = 4):
        # Wskazanie rozmiaru planszy (domyslnie 4)
        self.board_size = board_size
        # Utowrzenie planszy skladajacej sie z samych zer
        self.board = [[0]*board_size for i in range(board_size)]
        # Ustawienie wspolrzednych pustki na koncu (prawy dolny rog) planszy
        self.blank_position = (board_size-1,board_size-1)
        
        # Uzupelnienie planszy liczbami od 1 do 16-tu
        for i in range(board_size):
            for j in range(board_size):
                self.board[i][j] = i * board_size + j + 1
        
        # Umieszczenie pustki (zera) na koncu planszy (prawy dolny rog)
        self.board[self.blank_position[0]][self.blank_position[1]] = 0
        
        # Przetasowanie planszy
        self.shuffle()
        
    # Reprezentacja klasy "Puzzle" za uzyciem ciagu znakow (moznosc "printowania")
    def __str__(self):
        outStr = ''
        for i in self.board:
            outStr += '\t'.join(map(str,i))
            outStr += '\n'
        return outStr
    
    # Nadpisanie metody "getitem", aby moc wykorzystywac notacje "[]"
    def __getitem__(self, key):
        return self.board[key]
    
    # Przetasowanie planszy
    def shuffle(self):
        number_of_shuffles = 1000
        
        # Dla zadanej liczby tasowan:
        for i in range(number_of_shuffles):
            # wybierz losowo kierunek 
            direction = choice(self.DIRECTIONS)
            # wykonaj ruch
            self.move(direction)
    
    # Wykonanie ruchu
    def move(self, direction):
        # Nowa pozycja pustki = stara pozycja + przesuniecie o zadany kierunek
        new_blank_position = (self.blank_position[0] + direction[0],
                              self.blank_position[1] + direction[1])
        
        # Jezeli przesuniecie "wychodzi" poza plansze, zwroc blad
        if new_blank_position[0] < 0 or new_blank_position[0] >= self.board_size \
            or new_blank_position[1] < 0 or new_blank_position[1] >= self.board_size:
            return False
        
        # Zamiana pustki na zamieniany klocek
        self.board[self.blank_position[0]][self.blank_position[1]] = self.board[new_blank_position[0]][new_blank_position[1]]
        # Przypisanie pustki do nowego miejsca
        self.board[new_blank_position[0]][new_blank_position[1]] = 0
        # Nadpisanie pustki
        self.blank_position = new_blank_position
        return True 
 
    def check(self):
        # Sprawdzenie, czy liczby sa w odpowiedniej kolejnosci
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != i * self.board_size + j + 1 and self.board[i][j] != 0:
                    return False
                
        return True