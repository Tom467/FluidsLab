Above is a basic example of an acceptable table:

Each column is one parameter from the experiment. The first entry is the parameter's name. The parameter name supports LaTeX expressions (i.e. greek characters and subscripting). The second row contains the units for each parameter. Most common parameter types (i.e. velocity, mass, viscosity_dynamic, etc.) can be written directly or the units can be written explicitly. Note that everything will be converted into meters, kilograms, and seconds (support for temperature conversion will be added in the future and will convert to Kelvins).

The rest of the rows contain all the data collected from the experiment. All parameters should have the same number of recorded measurements, meaning there should not be any empty cells. Zero and NaN values are not acceptable (zero values lead to division by zero errors, and proper handling of NaN values has not been considered yet).

Notice column "A" in the image above is called "Label", this tells the software to ignore this column during analysis and just use it for plotting. The second cell in this column should be the name of the label. This column is optional.
