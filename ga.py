import typing, random, time, math
from typing import List

# TODO Augment
TASTES: List[str] = [
    "sour",
    "sweet", 
    "salty", 
    "bitter", 
    "herby",  
] 

# TODO Augment
INGREDIENT_FUNCTION: List[str] = [
    "base",
    "binder",
    "topping",
    "enhancer",
    "texture",
]


class Ingredient:
    """
    
    """
    def __str__(self):
        return f"{self.name}, {self.quantity} {self.unit}, {self.moistness}, {self.taste}, {self.function_}, {self.intensity}"

    def __init__(
            self,
            name:       str,
            quantity:   float,
            unit:       str,
            moistness:  float = -1,
            taste:      str = "",
            function_:  str = "",
            intensity:  float = -1,
        ):
        self.name: str = name
        self.quantity: float = quantity     # Numerical amount to add
        self.unit: str = unit               # Unit of measurement (g, l, kg, etc.)
        self.moistness: float = moistness   # Desert - [0:10] - Ocean
        self.taste: str = taste   # Salty, bitter, etc...
        self.function_: str = function_     # Function of ingredient
        self.intensity: float = intensity           # Weak taste - [0:10] - Strong taste


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
    mut_type = ["add", "del", "mut"]

    def __str__(self):
        recipe_str = ""
        for s in self.ingredients:
            recipe_str += str(s) + "\n"
        return recipe_str

    def __init__(
            self,
            ingredients: List[Ingredient],
            target_taste: str = None,
        ):

        self.ingredients: List[Ingredient] = ingredients

        # Select random taste target if none is given
        if target_taste: 
            self.target_taste: str = target_taste
        else:
            self.target_taste: str = random.choice(TASTES)



    def mutate(self, db: List[Ingredient], mutation_rate: float = 0.8):
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
        if mut_type == "add":
            done = False
            while not done:
                
                # Check for duplicate ingredient
                to_add = random.choice(db)
                done = True
                for i in self.ingredients:
                    if to_add.name == i.name:
                        done = False
                    
                
            self.ingredients.append(random.choice(db))
            
        # Perform deletion
        elif mut_type == "del":
            size = len(self.ingredients)
            if size > 1:
                self.ingredients.pop(random.randint(0, size - 1))

        # Perform point mutation on features
        elif mut_type == "mut":            
            # Decide amount of mutations to do
            mutations = random.gauss(0, 3)      # TODO tweak OR why not just do 1!
            mutations: int = abs(math.ceil(r))
            # size = len(self.ingredients)
            # if mutations > size:
                # mutations = size

            # Decide which ingredients to mutate and execute
            mutants = random.sample(self.ingredients, min(mutations, len(self.ingredients)))

            for i in mutants:
                i.mutate()


    def _moistness(self) -> float:
        """
        
        """
        # TODO Test
        moistness = 0      # Desert - [0:10] - Ocean
        # [print(type(i), i) for i in self.ingredients]
        parts = [(i.quantity / 1000, i.moistness) for i in self.ingredients]
        for p in parts:
            moistness += p[0] * p[1]
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
        ret = 0
        # TODO Calculate fitness score
        # Base it on:
        # Moistness
        ret += abs(5 - self._moistness())     # TODO Scaling
        # Good distribution of types (base, enhancer, binder, etc)
        # Cohesian of ingredient tastes according to recipe target taste
        # Quantity of ingredient paired with the ingredient's intensity

        return ret


class GA:
    def __init__(
            self,
            database: List[Ingredient]
        ):
        """
        
        """
        self.population: List[Recipe] = []
        self.database = database


    def _init_population(self, population_size: int):
        """
        
        """
        # TODO test
        population: list[Recipe] = []

        for _ in range(population_size):
            recipe_size = random.gauss(4, 0.75)     # TODO tweak
            recipe_size = math.ceil(recipe_size)
            ingredients = random.sample(self.database, max(recipe_size, 2))
            child = Recipe(ingredients)
            population.append(child)

        return population

    def _selection(
            self, 
            population: List[Recipe],
            selection_size: int,
        ) -> list[Recipe]:
        """

        """
        # TODO test
        # TODO Do we need ascending or decending????
        evals = [(recipe.evaluate() , recipe) for recipe in population]
        evals.sort(key=lambda tup: tup[0], reverse=False)
        return [x[1] for x in evals[:selection_size]]


    def _crossover(
            self, 
            population: List[Recipe], 
            population_size: int,
        ) -> list[Recipe]:
        """
        
        """
        # TODO test
        children: list[Recipe] = []

        for _ in range(population_size):
            # Sample recipes to combine
            recipe1, recipe2 = random.sample(population, 2)
            ingredients: List[Ingredient] = recipe1.ingredients + recipe2.ingredients
            random.shuffle(ingredients)

            # Decide amount of ingredients of child
            small = min(recipe1.length(), recipe2.length())
            large = max(recipe1.length(), recipe2.length())
            num_ingredients = random.randint(small, large)

            # Select ingredients
            ingredients = ingredients[0 : num_ingredients]
            
            # Select taste target
            # r = random.random()
            # if r > small_len / big_len:
            #     taste = parents[dom - 1].target_taste
            # else:
            #     taste = parents[dom - 1].target_taste
            if random.random() > 0.5:
                taste = recipe1.target_taste
            else:
                taste = recipe2.target_taste

            # Create child
            child = Recipe(ingredients, taste)
            children.append(child)
        
        return children


    def _mutation(
            self, 
            population: List[Recipe], 
            # db: List[Recipe],
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
    # [name, quantity, unit, taste, moistness, type]
    dataset = [
        ["water", 500, "ml", "neutral", 10, "moisture agent"],
        ["milk", 500, "ml", "creamy", 8, "moisture agent"],
        ["honey", 100, "ml", "sweet",4, "sweetener"],
        ["butter", 300, "g", "creamy", 8, "fat"],
        ["margarine", 300, "g", "creamy", 8, "fat"],
        ["sugar", 300, "g", "sweet", 2, "sweetener"],
        ["egg", 100, "g", "neutral", 7, "binding agent"],
        ["flour", 500, "g", "neutral", 1, "dry ingredient"],
        ["cocoa powder", 100, "g", "bitter", 1, "dry ingredient"],
        ["salt", 1, "g", "salty", 1, "seasoning"],
        ["walnuts", 50, "g", "nutty", 1, "mix-in"],
        ["coconut flakes", 100, "g", "tropical", 1, "mix-in"],
    ]

    dataset = [Ingredient(i[0], i[1], i[2], i[4], i[3], i[5]) for i in dataset]

    # Parameters that tune the genetic algorithm
    num_ingredients = 3
    population_size = 20    # Amount of created children each epoch (>2)
    selection_size  = 10    # Amount of selected children to advance (>2)
    mut_delta  = 1          # TODO Not sure if needed
    epochs = 100

    ga = GA(dataset)
    recipes = ga.run(epochs, population_size, selection_size, mut_delta)
    for recipe in recipes:
        print(recipe)
if __name__ == "__main__":
    main()