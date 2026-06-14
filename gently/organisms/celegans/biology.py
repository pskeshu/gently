"""
C. elegans developmental biology knowledge.

Prompt text describing embryonic development for inclusion in the
system prompt. This is the canonical source — prompts.py imports from here.
"""

BIOLOGY_KNOWLEDGE = """
# C. elegans Embryonic Development

C. elegans embryogenesis is highly stereotyped and invariant, proceeding through
well-defined stages:

## Key Developmental Stages

1. **One-cell stage (0-40 min)**: Fertilized egg with asymmetric first division
   - Anterior-posterior axis established
   - P granules segregate to posterior

2. **2-cell stage (~40-55 min)**: Unequal division into AB (anterior) and P1 (posterior)
   - AB larger, divides first
   - P1 smaller, divides ~2 min after AB

3. **4-cell stage (~55-80 min)**: AB divides into ABa/ABp, P1 into EMS/P2
   - Characteristic diamond shape
   - Cell fate determination begins

4. **8-cell stage (~80-105 min)**: Continued divisions
   - EMS divides into MS and E (gut precursor)
   - P2 divides into C and P3

5. **Gastrulation (~210 min)**: Internalization of cells
   - E cells (gut) move inward
   - Embryo begins elongation

6. **Comma stage (~400 min)**: Embryo curves into comma shape
   - Major morphogenesis
   - Organ systems forming

7. **1.5-fold stage (~450 min)**: Elongation continues
   - Embryo 1.5x length of eggshell

8. **2-fold stage (~500 min)**: Further elongation
   - Embryo 2x length, begins folding

9. **3-fold stage (~550 min)**: Near full elongation
   - Embryo 3x length, tightly folded
   - Movement begins

10. **Hatching (~800 min, 13-14 hours at 20°C)**: L1 larva emerges
    - Breach of eggshell (vitelline membrane)
    - Active pushing and wriggling
    - Takes 5-30 minutes to fully emerge

## Observable Features for AI Analysis

- **Cell division timing**: Precise intervals between divisions
- **Cell positions**: Stereotyped spatial arrangement
- **Eggshell integrity**: Clear boundary until hatching
- **Morphology changes**: Spherical → comma → elongated
- **Movement**: Increases dramatically after 3-fold stage
- **Hatching**: Visible breach, emerging larva

## Temperature Dependence

Development rate is temperature-dependent:
- 20°C: ~14 hours to hatching (standard)
- 25°C: ~10 hours to hatching (faster)
- 15°C: ~24 hours to hatching (slower)

## Common Phenotypes to Detect

- **Normal development**: Follows timeline above
- **Delayed**: Slower progression through stages
- **Arrest**: Development stops at specific stage
- **Abnormal morphology**: Incorrect cell divisions, elongation defects
- **Death**: Loss of cell boundaries, cytoplasmic blebbing
"""
