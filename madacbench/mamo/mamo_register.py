class Singleton:
    def __new__(cls, *arg, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class Task(Singleton):
    @staticmethod
    def get_task(name="all"):
        if "all" in name:
            nobjs = [3, 5, 7]
            if name != "all":
                nobjs = [int(name[-1])]
            DTLZ = ["DTLZ" + str(i) + "_" + str(j)
                    for i in [2, 4] for j in nobjs]
            WFG = ["WFG" + str(i) + "_" + str(j)
                   for i in range(4, 10) for j in nobjs]
            task = DTLZ + WFG
            return task


MAENV_REGISTER = {
    "M_2_46_3": [["DTLZ2", "WFG4", "WFG6"], [3]],
    "M_2_46_5": [["DTLZ2", "WFG4", "WFG6"], [5]],
    "M_2_46_7": [["DTLZ2", "WFG4", "WFG6"], [7]],
    "M_2_46_357": [["DTLZ2", "WFG4", "WFG6"], [3, 5, 7]],
}


def get_maenv(key):
    if key in MAENV_REGISTER:
        return MAENV_REGISTER[key]
    else:
        key = key.split("_")
        key[0] = [key[0]]
        key[1] = [int(key[1])]
        return key


if __name__ == "__main__":
    key = "WFG6_3"
    print(get_maenv(key))
    key = "WFG45_3"
    print(get_maenv(key))
    key = "M245_235"
    print(get_maenv(key))
