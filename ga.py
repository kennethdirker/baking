import typing, random, time, math
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
    Class that represents a cookie recipe.
    """

    # Ingredient mutation types
    mut_type = {"add", "del", "mut"}

    def __init__(
            self,
            ingredients: list[Ingredient],
            target_taste: str = None,
        ):

        self.ingredients: list[Ingredient] = ingredients

        # Select random taste target if none is given
        if target_taste: 
            self.target_taste: str = target_taste
        else:
            self.target_taste: str = random.choice(TASTES)



    def mutate(self, db: list[Ingredient], mutation_rate: float = 0.8):
        # Are we going to mutate?
        r: float = random.random()
        if r < mutation_rate:
            return 
        
        # What kind of mutation will take place?
        mut_type: str = self.mut_type[random.randint(0, 2)]

        # Perform addition
        if mut_type is "add":
            self.ingredients.append(ingredient)
            
        # Perform deletion
        elif muyt_type is "del":
            size = len(self.ingredients)
            if size > 0:
                self.ingredients.pop(random.randint(0, size - 1))

        # Perform point mutation on features
        elif mut_type is "mut":            
            # Decide amount of mutations to do
            mutations = random.gauss(0, 3)
            mutations: int = abs(math.ceil(r))
            size = len(self.ingredients)
            if mutations > size:
                mutations = size

            # Decide which ingredients to mutate and execute
            mutants = random.sample(range(self.ingredients), mutations)
            for i in mutants:
                self.ingredients[i].mutate()


    def moistness(self) -> float:
        """
        
        """
        moistness = -1      # Desert - [0:100] - Ocean
        return moistness


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
    def __init__(
            self,
            database: list[Ingredient]
        ):
        """
        
        """

        self.population: list[Recipe] = []

    def _selection(
            self, 
            population: list[Recipe],
            selection_size: int,
        ) -> list[Recipe]:
        """
        
        """

        # TODO do we need ascending or decending????
        evals = [(recipe.evaluate() , recipe) for recipe in population]
        evals.sort(key=lambda tup: tup[0], reverse=True)
        return evals[:selection_size]

    def _crossover(
            self, 
            population: list[Recipe], 
            pop_size: int,
        ) -> list[Recipe]:
        """
        
        """
        children: list[Recipe] = []

        for _ in range(pop_size):
            # Sample recipes to combine
            recipe1, recipe2 = random.sample(population, 2)
            ingredients = recipe1.ingredients + recipe2.ingredients
            random.shuffle(ingredients)

            # Decide amount of ingredients of child
            len1 = recipe1.length()
            len2 = recipe2.length()
            parents: list[Recipe] = [recipe1, recipe2]
            dom = 0        # Flag indicating the bigger recipe

            r = random.random()
            if r > 0.5:
                dom = 1
                small_len = len1
                big_len   = len2
            else:
                small_len = len2
                big_len   = len1

            # Select ingredients
            ingredients = ingredients[0:new_len-1]
            
            r = random.random()
            if r > small_len / big_len:
                taste = parents[len2_dom].target_taste()
            else:
                taste = parents[len2_dom].target_taste()

            # Create child
            child = Recipe(ingredients, taste)
            children.append(child)
        
        return children


    def _mutation(
            self, 
            population: list[Recipe], 
            db: list[Recipe],
            mutation_rate: float = 0.8
        ) -> None:
        """
        
        """
        for recipe in population:
            recipe.mutate(db, mutation_rate)
        


    def _normalize(self):
        """
        
        """

        
    def run(
            self, 
            epochs: int = 100,
            pop_size: int = 20,
            sel_size: int = 10,
            mut_delta: float = 1,
        ) -> list[Recipe]:
        """
        
        """
        # TODO is mut_delta needed???
        print(
            f"Starting GA with: {epochs} epochs, {pop_size} individuals, " \
            f"{sel_size} selected, delta: {mut_delta}."
        )

        # Timers
        start_time = time.time()
        epoch_time = start_time

        # Initialize population
        population = ...

        # Run algorithm
        for i in range(epochs):
            # TODO Select
            # TODO Crossover
            # TODO Mutation
            # TODO Normalization

            if i % 10 == 0:
                new_time = time.time()
                print(
                    f"Epoch {i}, Epoch time taken: {new_time - epoch_time}, " \
                    f"Total time taken: {new_time - start_time}"
                )
                epoch_time = new_time
        
        print(
            f"\nFinished:\n\tTotal time taken: {time.time() - start_time}\t" \
            f"Number of epochs: {epochs}"
        )


def main():
    # TODO Initialize dataset of ingredients
    dataset = ...

    # Parameters that tune the genetic algorithm
    num_ingredients = 6
    population_size = 20    # Amount of created children each epoch
    selection_size  = 10    # Amount of selected children to advance
    mut_delta  = 1          # TODO Not sure if needed

    ga = GA(dataset, population_size, selection_size, mutation_delta)



if __name__ == "__main__":
    main()