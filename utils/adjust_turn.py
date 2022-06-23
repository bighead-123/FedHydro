
class AdjustTurn:
    def __init__(self, batches):
        self.__local_turn = 0
        self.__TURN = batches

    def set_turn(self, turn):
        self.__TURN = turn

    def set_local_turn(self):
        self.__local_turn += 1

    def get_local_turn(self):
        return self.__local_turn

    def get_turn(self):
        return self.__TURN

    def is_end(self):
        return self.__local_turn >= self.__TURN

    def clear_turn(self):
        self.__local_turn = 0
