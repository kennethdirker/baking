import typing, random
# Dataset with:
#   



# Evaluation:
#   Ratios
#   Taste profile of ingredients
#   Roles of ingredients (binder, enhancer, etc...)
#   

# TODO Augment
TASTES: list[str] = [
    "sour",
    "sweet", 
    "salty", 
    "bitter", 
    "herby",  
] 

INGREDIENT_FUNCTION: list[str] = [
    "base",
    "binder",
    "topping",
    "enhancer",
    "texture",
]


class Ingredient:
    """
    
    """

    def __init__(
            self,
            name:       str,
            quantity:   float,
            unit:       str,
            moistness:  float = -1,
            taste_type: str = "",
            function_:  str = "",
            power:      float = -1,
        ):

        self.name: str = name
        self.quantity: float = quantity     # Numerical amount to add
        self.unit: str = unit               # Unit of measurement (g, l, kg, etc.)
        self.moistness: float = moistness   # Desert - [0:100] - Ocean
        self.taste_type: str = taste_type   # Salty, bitter, etc...
        self.function_: str = function_     # Function of ingredient
        self.power: float = power           # Weak taste - [0:100] - Strong taste


    def mutate(self, delta: float = 1):
        """
        
        """
        random_factor = random.gauss(1, 0.05)    # TODO experiment with standard deviation
        self.quantity = self.quantity ** random_factor
        


class Recipe:
    """
    
    """

    def __init__(
            self,
            target_taste: str = None,
            moistness: float = -1,
        ):

        self.ingredients: list = []


        # Select random taste target if none is given
        if target_taste: 
            self.target_taste: str = target_taste
        else:
            self.target_taste: str = random.choice(TASTES)

        self.moistness: float = -1   # Desert - [0:100] - Ocean


    def add_ingredient(self, ingredient: Ingredient):
        """
        
        """
        self.ingredients.append(ingredient)


    def remove_ingredient(self):
        """
        
        """
        size = len(self.ingredients)
        if size > 0:
            self.ingredients.pop(random.randint(0, size - 1))


    def length(self) -> int:
        """
        
        """
        return len(self.ingredients)


    def evaluate(self) -> float:
        """
        
        """
        ret = -1
        return ret


class GA:
    def __init__(self):
        """
        
        """
        ...

    def _selection(self):
        """
        
        """


    def _crossover(self):
        """
        
        """


    def _mutation(self):
        """
        
        """


    def _normalize(self):
        """
        
        """

        
    def run(self):
        """
        
        """
        # TODO Select
        # TODO Crossover
        # TODO Mutation
        # TODO Normalization



def main():
    # TODO Initialize dataset of ingredients
    dataset = ...

    # Parameters that tune the genetic algorithm
    population_size = 20    # Amount of created children each epoch
    selection_size  = 10    # Amount of selected children to advance
    mutation_delta  = 1     

    ga = GA()



if __name__ == "__main__":
    main()