import random
import sys

# Truth table mapper for the logic gates.
# Key is the index as below which has the a dictionary object.
# Each dict has a tuple as key denoting (x1, x2) and value has y^ value for that combination. 
# 1. XOR 
# 2. AND
# 3. OR
truth_table = {
    1: {
        (1, 1): 0,
        (0, 1): 1,
        (1, 0): 1,
        (0, 0): 0,
    },
    2: {
        (1, 1): 1,
        (0, 1): 0,
        (1, 0): 0,
        (0, 0): 0,
    },
    3: {
        (1, 1): 1,
        (0, 1): 1,
        (1, 0): 1,
        (0, 0): 0,
    },
}


def activation(x):
    return 1 if x >= 0 else 0


def randomize_inps_float() -> list[float]:
    """
    Return the parameters for check_nn function.
    Note: All values are in list are floats. Useful for better precision calculation.
    """
    return [round(random.uniform(-5, 5), 2) for i in range(9)]


def randomize_inps_int() -> list[int]:
    """
    Return the parameters for check_nn function.
    Note: All values are in list are integer. Want more precision? use 'randomize_inps_float'.
    """
    return [random.randint(-5, 5) for i in range(9)]


def check_nn(case, x1h1, x2h1, x1h2, x2h2, w_h1, w_h2, h1y, h2y, w_y) -> bool:
    """
    Returns whether the input weights and biases are valid wrt neural net function f(x1, x2) -> y^. 
    """
    for value in case.keys():
        x1, x2 = value
        h1 = (x1 * x1h1) + (x2 * x2h1) + (1 * w_h1)
        h1 = activation(h1)

        h2 = (x1 * x1h2) + (x2 * x2h2) + (1 * w_h2)
        h2 = activation(h2)

        y = (h1 * h1y) + (h2 * h2y) + (1 * w_y)
        y = activation(y)

        if case[value] != y:
            return False
    return True


if __name__ == "__main__":
    try:
        category = int(input("Select an option below:\n1. Test your work\n2. Generate values\n"))
        case_ = -1
        if category == 1:
            inps = map(float, input('Enter values for x1h1, x2h1, x1h2, x2h2, w_h1, w_h2, h1y, h2y, w_y\n').split(' '))
        elif category == 2:
            print("""Select logic gate.\n1. XOR\n2. AND\n3. OR""")
            case_ = int(input())
            inps = randomize_inps_int()
        else:
            raise ValueError('Try again with available options.')
        count_ = 0
        if len(sys.argv) == 2 and sys.argv[1]:
            count_ = int(sys.argv[1])
        else:
            count_ = 10
        
        group = list()
        i = 0

        print(check_nn(truth_table[case_], *inps))

        while i < count_:
            inps = randomize_inps_int()
            res = check_nn(truth_table[case_], *inps)
            if res:
                group.append(inps)
                i += 1
        [print(g) for g in group]
    except ValueError:
        print('Try again. Available options (1 or 2)')

