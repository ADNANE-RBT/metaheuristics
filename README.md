# metaheuristics
# Nature-Inspired Optimization Algorithms

This project implements two nature-inspired optimization algorithms: Moth Flame Optimization (MFO) and Honey Badger Optimization (HBO). Both algorithms are designed to solve complex optimization problems by mimicking behaviors observed in nature.

## Moth Flame Optimization (MFO)

MFO is inspired by the navigation method of moths in nature, called transverse orientation. Moths fly at night by maintaining a fixed angle with respect to the moon, an effective technique for traveling in a straight line over long distances.

### Key Features of MFO:

1. **Population**: Consists of moths and flames.
2. **Movement**: Moths update their positions with respect to flames using a spiral flying path.
3. **Adaptive Flame Number**: The number of flames decreases over iterations, intensifying the exploitation phase.
4. **Spiral Update**: Uses a logarithmic spiral to define the moth's flying path around flames.

## Honey Badger Optimization (HBO)

HBO is inspired by the foraging behavior and aggressive nature of honey badgers. These animals are known for their fearlessness and ability to adapt to various environments.

### Key Features of HBO:

1. **Population**: Consists of honey badgers.
2. **Behavior**: Alternates between foraging and aggressive behaviors.
3. **Movement**: Updates positions based on the best solution found (foraging) or a random member of the population (aggressive).
4. **Simplicity**: Uses a straightforward update rule without complex mathematical functions.

## Comparison between MFO and HBO

1. **Inspiration Source**:
   - MFO: Based on moth navigation using moonlight.
   - HBO: Based on honey badger foraging and aggressive behaviors.

2. **Population Structure**:
   - MFO: Uses two sets of solutions (moths and flames).
   - HBO: Uses a single set of solutions (honey badgers).

3. **Update Mechanism**:
   - MFO: Uses a logarithmic spiral path for updates.
   - HBO: Uses linear position updates based on best solution or random population member.

4. **Adaptive Parameters**:
   - MFO: Adapts the number of flames over iterations.
   - HBO: Does not have adaptive parameters.

5. **Complexity**:
   - MFO: More complex due to spiral paths and flame number adaptation.
   - HBO: Simpler implementation with straightforward update rules.

6. **Exploration vs Exploitation**:
   - MFO: Balances exploration and exploitation through flame number reduction.
   - HBO: Balances through alternating between foraging (exploitation) and aggressive (exploration) behaviors.

## Usage

Both algorithms are implemented to work with various fitness functions. To use:

1. Ensure you have `numpy` and `matplotlib` installed.
2. Place `fitness_functions.py` and `report_generator.py` in the same directory as the main scripts.
3. Run either `moth_flame_optimization.py` or `honey_badger_optimization.py`.
4. The algorithms will generate reports for each fitness function.
5. You'll be prompted if you want to see visualizations of the optimization process.

## Conclusion

Both MFO and HBO offer unique approaches to optimization inspired by nature. MFO provides a more structured approach with its spiral paths and adaptive flame number, potentially making it suitable for problems requiring a balance of exploration and exploitation. HBO, with its simpler implementation, might be more adaptable to a wider range of problems and easier to tune.