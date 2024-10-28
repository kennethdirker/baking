import typing, random, time, math, json
from typing import List

TASTES: List[str] = [
    "bitter",
    "chocolate", 
    "coffee", 
    "creamy", 
    "floral",
    "fresh",
    "fruity",
    "mealy",
    "neutral",
    "nutty",
    "salty",
    "spicy",
    "sweet",
    "tangy",
    "tropical"  
] 

INGREDIENT_FUNCTIONS: dict[str, int] = {
    "fat": 3,
    "sweetener": 3,
    "binding agent": 4,
    "flavoring": 5,
    "dry ingredient": 1,
    "mix-in": 3,
    "raising agent": 5,
    "moisture agent": 2
}


class Ingredient:
    """
    Class that represents an ingredient in a recipe.
    """
    def __str__(self):
        return f"{self.name}, {float(self.quantity):.4} {self.unit}, {self.moistness}, {self.taste}, {self.function_}, {self.intensity}"

    def __init__(
            self,
            name:       str,
            quantity:   float = -1,
            unit:       str = "g",
            moistness:  float = -1,
            taste:      List[str] = "",
            function_:  str = "",
            intensity:  float = -1,
        ):
        self.name: str = name
        if quantity == -1:
            self.quantity: float = random.randint(1, 1000)
        else:
            self.quantity: float = quantity # Numerical amount to add
        self.unit: str = unit               # Unit of measurement (g, l, kg, etc.)
        self.moistness: float = moistness   # Desert - [0:10] - Ocean
        self.taste: str = taste             # Salty, bitter, etc...
        self.function_: str = function_     # Function of ingredient
        self.intensity: float = intensity   # Weak taste - [0:10] - Strong taste


    def mutate(self, delta: float = 1):
        """
        Mutate the quantity of the ingredient.
        """
        if self.quantity > 0:
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


    def merge_ingredients(self):
        """
        Merge doubly listed ingredients into one ingredient by
        aggregating the ingredient quantities.
        """
        d = {}
        merged_ingredients = []
        
        # Merge ingredients in dict
        for i in self.ingredients:
            if i.name in d:
                d[i.name].quantity += i.quantity
            else:
                d[i.name] = Ingredient(
                    i.name, 
                    i.quantity,
                    i.unit,
                    i.moistness,
                    i.taste,
                    i.function_,
                    i.intensity
                )

        # Create new list of ingredients
        for value in d.values():
            merged_ingredients.append(value)
        
        self.ingredients = merged_ingredients


    def mutate(self, db: List[Ingredient], mutation_rate: float = 0.8):
        """
        Potentially mutate the recipe by adding/removing an ingredient or
        changing the quantity of ingredients.
        """
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

            # Decide which ingredients to mutate and execute
            mutants = random.sample(self.ingredients, min(mutations, len(self.ingredients)))

            for i in mutants:
                i.mutate()


    def _moistness(self) -> float:
        """
        Calculate how moist the resulting mesh of ingredients might be.

        Return a moistness score in the range: Desert - [0:10] - Ocean
        """
        return sum([i.quantity * i.moistness / 1000 for i in self.ingredients])

    def _intensity(self) -> float:
        """
        Calculate how intense the taste of the resulting mesh of ingredients 
        might be.

        Return a intensity score in the range: Water - [0:20] - very very strong
        """
        return sum([i.quantity * i.intensity / 1000 for i in self.ingredients])


    def length(self) -> int:
        return len(self.ingredients)
    

    def normalize(self) -> None:
        """
        Normalize the recipe to contain 1000g of ingredients.
        """
        # NOTE: Assumes that all measures are in grams/milliliters!!!!
        # Calculate total mass of recipe
        sum_: float = 0      # In grams/milliliters
        for ingredient in self.ingredients:
            sum_ += ingredient.quantity

        # Scale ingredients to match "1000 g" of mass
        for ingredient in self.ingredients:
            ingredient.quantity = (1000 * ingredient.quantity) / sum_ 


    def sort(self):
        self.ingredients.sort(key = lambda ingredient: ingredient.name)
            


    def evaluate(self, verbose = False) -> float:
        """
        Calculate the fitness of the recipe. The fitness is a positive,
        where a good recipe closes to 0. Evaluation is based on:

        Moistness:
        The resulting mesh from combining the different
        recipes should contain the right amount of water.

        Function:
        Penalize not using all types of ingredients.

        Intensity
        We want balanced tastes. So no extremely salty cookies!


        Returns the fitness as a positive float.
        """ 
        # Count which functions are present in the recipe
        functions = {}
        for i in self.ingredients:
            if i.function_ not in functions:
                functions[i.function_] = True

        m = self._moistness() 
        i = self._intensity()

        fitness = 0
        fitness += 2 ** abs(self._moistness() - 4.5)    # TODO Scaling
        fitness += 4 ** abs(self._intensity() - 4)    # TODO Scaling
        fitness += 2 ** (len(INGREDIENT_FUNCTIONS) - len(functions))

        if verbose:
            print(f"Moistness: {m}, Intensity: {i}")
        return fitness


class GA:
    """
    Class responsible for executing a genetic algorithm to discover cookie
    recipes!
    """
    def __init__(
            self,
            database: List[Ingredient]
        ):
        self.population: List[Recipe] = []
        self.database = database


    def _init_population(self, population_size: int, num_ingredients: int):
        """
        Initialize a population of recipes. Recipes are given random
        ingredients, where the amount of ingredients has a gaussian variance
        around the $num_ingredients parameter. Ingredient quantities are
        randomly assigned a value in the range [1:1000].

        Returns a list of initialized recipes.
        """
        population: list[Recipe] = []

        for _ in range(population_size):
            recipe_size = random.gauss(num_ingredients, 0.75)
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
        Select the fittest recipes from the population, 
        according to the evaluation function.

        Returns a list with the fittest recipe population.
        """
        evals = [(recipe.evaluate() , recipe) for recipe in population]
        evals.sort(key=lambda tup: tup[0], reverse=False)
        return [x[1] for x in evals[:selection_size]]


    def _crossover(
            self, 
            population: List[Recipe], 
            population_size: int,
        ) -> list[Recipe]:
        """
        Create a new population of new recipes by sampling ingredients
        from the 2 parents.

        Returns a list containing the new population.
        """
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
            mutation_rate: float = 0.8
        ) -> None:
        """
        Mutate a recipe by adding/removing ingredients or adjusting
        the quantity of ingredients used.

        This function is applied in place to the supplied population list.
        """
        for recipe in population:
            recipe.mutate(self.database, mutation_rate)
        


    def _normalize(self, population: list[Recipe]) -> None:
        """
        Normalize all recipes to each weigh 1 kg.

        This function is applied in place to the supplied population list.
        """
        for recipe in population:
            recipe.normalize()


    def sort_recipes(self, population: List[Recipe]):
        """
        Sort ingredient list of the recipes alphabetically.

        This function is applied in place to the supplied population list.
        """
        for recipe in population:
            recipe.sort()


    def print_recipes(self, recipes: Recipe):
        for r in recipes:
            print(f"fitness: {r.evaluate(verbose=True):.4}")
            print(r)


    def run(
            self, 
            epochs: int = 100,
            population_size: int = 1,
            selection_size: int = 1,
            num_ingredients: int = 3
        ) -> list[Recipe]:
        """
        Run the genetic algorithm.

        Return a list of recipes.
        """
        print(
            f"Starting GA with: {epochs} epochs, {population_size} recipes, " \
            f"{selection_size} selected."
        )

        # Timers
        start_time = time.time()
        epoch_time = start_time

        # Initialize population
        population = self._init_population(population_size, num_ingredients)
        self._normalize(population)

        # Run algorithm
        for i in range(epochs):
            population = self._selection(population, selection_size)

            if population_size > 1:
                population = self._crossover(population, population_size)

            # Merge duplicate ingredient listings
            for recipe in population:
                recipe.merge_ingredients()

            self._mutation(population)
            self._normalize(population)
            self.sort_recipes(population)

            if i + 1 % 10 == 0:
                new_time = time.time()
                print(
                    f"Epoch {i}, Epoch time taken: {new_time - epoch_time:.4}, " \
                    f"Total time taken: {new_time - start_time:.4}"
                )
                epoch_time = new_time
        
        print(
            f"\nFinished:\n\tTotal time taken: {time.time() - start_time:.4}\t" \
            f"Number of epochs: {epochs}\n"
        )

        # Select final recipes
        population = self._selection(population, selection_size)
        self.print_recipes(population)
        return population

def main():
    with open("ingredients_v3.json") as json_file:
        data = json.load(json_file)
        dataset = []

        # Initialize dataset
        for i in data["cookie_ingredients"]:
            dataset.append(Ingredient(
                name =      i["name"],
                moistness = i["moisture_level"],
                taste =     i["taste"],
                function_ = i["type"],
                intensity = i["intensity"]
            ))

        # Parameters that tune the genetic algorithm
        num_ingredients = 5
        population_size = 100    # Amount of created children each epoch (>2)
        selection_size  = 20    # Amount of selected children to advance (>2)
        epochs = 100

        ga = GA(dataset)
        recipes = ga.run(epochs, population_size, selection_size)

if __name__ == "__main__":
    main()