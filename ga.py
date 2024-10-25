import typing, random, time, math

# TODO Augment
TASTES: list[str] = [
    "sour",
    "sweet", 
    "salty", 
    "bitter", 
    "herby",  
] 

# TODO Augment
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
            intensity:      float = -1,
        ):

        self.name: str = name
        self.quantity: float = quantity     # Numerical amount to add
        self.unit: str = unit               # Unit of measurement (g, l, kg, etc.)
        self.moistness: float = moistness   # Desert - [0:100] - Ocean
        self.taste_type: str = taste_type   # Salty, bitter, etc...
        self.function_: str = function_     # Function of ingredient
        self.intensity: float = intensity           # Weak taste - [0:100] - Strong taste


    def mutate(self, delta: float = 1):
        """
        
        """
        # TODO test
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
        """
        
        """
        # TODO test
        # Are we going to mutate?
        r: float = random.random()
        if r < mutation_rate:
            return 
        
        # What kind of mutation will take place?
        mut_type: str = self.mut_type[random.randint(0, 2)]

        # Perform addition
        if mut_type is "add":
            self.ingredients.append(random.choice(db))
            
        # Perform deletion
        elif mut_type is "del":
            size = len(self.ingredients)
            if size > 0:
                self.ingredients.pop(random.randint(0, size - 1))

        # Perform point mutation on features
        elif mut_type is "mut":            
            # Decide amount of mutations to do
            mutations = random.gauss(0, 3)      # TODO tweak OR why not just do 1!
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
        # TODO test
        moistness = -1      # Desert - [0:100] - Ocean
        # TODO Calc moistness from ingredients
        return moistness


    def length(self) -> int:
        """
        
        """
        return len(self.ingredients)
    

    def _normalize(self) -> None:
        """
        
        """
        # TODO test
        # NOTE: Assumes that all measures are in grams/milliliters!!!!
        # Calculate total mass of recipe
        sum: float = 0      # In grams/milliliters
        for ingredient in self.ingredients:
            sum += ingredient.quantity

        # Scale ingredients to match "1000 g" of mass
        for ingredient in self.ingredients:
            ingredient.quantity = (1000 * ingredient.quantity) / sum 
            


    def evaluate(self) -> float:
        """
        
        """
        # TODO test
        ret = -1
        # TODO Calculate fitness score
        # Base it on:
        # Moistness
        # Good distribution of types (base, enhancer, binder, etc)
        # Cohesian of ingredient tastes according to recipe target taste
        # Quantity of ingredient paired with the ingredient's intensity
        return ret


class GA:
    def __init__(
            self,
            database: list[Ingredient]
        ):
        """
        
        """
        self.population: list[Recipe] = []


    def _init_population(self, population_size: int):
        """
        
        """
        # TODO test
        population: list[Recipe] = []

        for _ in range(population_size):
            recipe_size = random.gauss(4, 0.75)     # TODO tweak
            ingredients = [random.sample(self.database) for _ in range(recipe_size)]
            child = Recipe(ingredients)
            population.append(child)

        return population

    def _selection(
            self, 
            population: list[Recipe],
            selection_size: int,
        ) -> list[Recipe]:
        """

        """
        # TODO test
        # TODO Do we need ascending or decending????
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
        # TODO test
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
            ingredients = ingredients[0 : small_len - 1]
            
            r = random.random()
            if r > small_len / big_len:
                taste = parents[dom].target_taste
            else:
                taste = parents[dom].target_taste

            # Create child
            child = Recipe(ingredients, taste)
            children.append(child)
        
        return children


    def _mutation(
            self, 
            population: list[Recipe], 
            # db: list[Recipe],
            mutation_rate: float = 0.8
        ) -> None:
        """
        
        """
        for recipe in population:
            recipe.mutate(self.database, mutation_rate)
        


    def _normalize(self, population: list[Recipe]) -> None:
        """
        
        """
        # TODO test
        for recipe in population:
            recipe._normalize()
        
    def run(
            self, 
            epochs: int = 100,
            population_size: int = 20,
            selection_size: int = 10,
            mut_delta: float = 1,
        ) -> list[Recipe]:
        """
        
        """
        # TODO test
        # TODO is mut_delta needed???
        print(
            f"Starting GA with: {epochs} epochs, {population_size} recipes, " \
            f"{selection_size} selected, delta: {mut_delta}."
        )

        # Timers
        start_time = time.time()
        epoch_time = start_time

        # Initialize population
        population = self._init_population(population_size)

        # Run algorithm
        for i in range(epochs):
            population = self._selection(population, selection_size)
            population = self._crossover(population, population_size)
            self._mutation(population)
            self._normalize(population)

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

        # Select final recipes
        return self._selection(population, population_size)


def main():
    # TODO Initialize dataset of ingredients
    dataset = ...

    # Parameters that tune the genetic algorithm
    num_ingredients = 6
    population_size = 20    # Amount of created children each epoch (>2)
    selection_size  = 10    # Amount of selected children to advance (>2)
    mutation_delta  = 1          # TODO Not sure if needed

    ga = GA(dataset, population_size, selection_size, mutation_delta)



if __name__ == "__main__":
    main()