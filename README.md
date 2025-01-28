**FluidsLab**
This software package automatically processes experimently collected data using Buckingham Pi Theory in order to display in chart format all possible correlations between experimental variables or parameters.

There are various other files in this repository which are planned for future development, but currently the main feature of the software is a ''dimensional analysis app'' which enables one to input a data file and automatically generate plots of scalings.  In Section 1, we give an overview of why such an app is useful for experimentalists and explain some of the associated theory of dimensional analysis.  In Section 2, we provide instructions how to run the app using the files in the repository.

1. **Overview**

**Experimental Research**
Every experiment has two types of variables: independent and dependent. The goal of the experimenter is to derive a relationship whereby they can determine the dependent variable by controlling independent variables. Sometimes this relationship can be directly derived from physics and governing equations, but other times the relationship is harder to uncover. The simplest of procedures for finding this relationship, or even validating a derived relationship, would look something like the following:

Define dependent variable
Define and list possible independent variables
Setup an experiment where the independent variables are varied
Measure the dependent variable

While this process is very simple and can produce powerful results, it turns out to be an "expensive" process in time, and resources. This is due to the exponential increase in combination with the increase of independent variables. For example, let's say we want to find a correlation between how many times a ball on a string will swing back and forth in a set amount of time. The number of swings is the dependent variable (what we don't control) and the set time is the independent variable (what we do control). We can run ten experiments changing how long we let the ball swing for each time and record the number of oscillations. We would unsurprisingly find that the longer we let the ball swing the more oscillations it will make. This is simple enough, but now suppose we also want to know how the mass of the ball, length of the string, release angle, and force of gravity affect the number of oscillations. If we were to follow the same approach and try all the different combinations, we would have to run 100,000 experiments (10 raised to the power of the number of independent variables). This many trial runs even for a simple experiment is clearly problematic, let alone for a more complex setup.

What if we could reduce the number of experiments we have to run by taking advantage of connections between the physical natures of the dependent variable: e.g. the effect due to the mass of the ball and the force of gravity seem likely to be related. These relationships can be defined with dimensional analysis which is done with the units of the variables (kg for mass, m/s^2 for gravity, etc.). One well know method of dimensional analysis is the Buckingham Pi Theorem.

**Using the Buckingham Pi Theorem for Simplified Analysis**
The Buckingham Pi Theorem outlines five steps for reducing the complexity of an experiment through dimensional analysis:

Write down the units of all the independent and dependent variables
Select the appropriate number of repeating variables:
The number of repeating variable depends on the number of base units included the units of the variables
Every base unit mass be included in at least one repeating variable
The units of the repeating variables must be linearly independent
Create a "Pi Group" by taking a non-repeating variable and cancelling its units by multiplying it with the repeating variables raised to exponential powers
Repeat the previous step until one pi group has been made for each non-repeating variable
The pi groups can now be used to find the correlation between the dependent and independent variable
Returning to our example of the ball on the string we would get the following:

Number of oscillations (has no units), Swing time (s), mass (kg), string length (m), release angle (radians), gravity (m/s^2)
We have three base units (kg, m, s) so we need three repeating variables. There are multiple possible combinations of repeating variables that would be acceptable, and we will return to a discussion on this later but for now let us select the period, gravity, and mass.
The length of the string has units of 'm' which can be cancelled out by period^-2 and gravity^1
As it turns out both the oscillations and the release angle are already dimensionless so this step is complete
Our final pi groups are Oscillations, release angle, and string length divided by the period squared times gravity
It is apparent now that instead of describing the experiment with one dependent and five independent variables, we can describe it with three pi groups. What originally needed 100,000 experiments can now be done with 100 experiments, a much more reasonable amount.

Returning to the topic of repeating variables, it was noted that there generally are several acceptable options. While any set of repeating variables would have the same effect in respect to simplification of the experiment, they will not result in the same relationships between pi groups. THe different repeating variable sets can be thought of as different coordinate systems. And just like how the equation for a unit circle is simpler in polar coordinates (r=1) than rectangular coordinates (x^2+y^2=1), so to some repeating variables give a simpler relationship than others.

**The Benefit of Automation**
Dimensional analysis has traditionally been done by hand by the experimenter. The process can be laborious and algebraic errors can occur in rushed calculations. It can be extra tedious exploring all the different combinations of repeating variables. The algorithmic nature of dimensional analysis suggests that a computer can aid in the process, and that is the goal of this project. This package seeks to take experimentally collected data and display all possible relations using dimensional analysis for the user to then interpret. Incredible data is useless if it is displayed ineffectively.


2. **Instructions for Use**

   As mentioned in the Introduction, there are currently a large number of files currently in the repository (some of them experimental features which have not yet been developed), but use of the dimensional analysis app only involves a few of the files.  The user should first install conda with Python and install the libraries in requirements.txt.  The following files then need to be cloned from the FluidsLab repository to the user's IDE:

FluidsLab_interface.py, dimensional_analysis.py, dimensional_matrix.py, pi_group.py, data_reader.py, fluids.py, parameter.py, plotter.py, units.py, util.py, convert.py

After this, the user should enter the command

streamlit run FluidsLab_interface.py

into the terminal.  This will bring up the Dimensional Analysis app with its own user interface.  Data files can then be inputted to generate plots as long as the data file is formatted correctly.  The first row of the data table must be filled with each parameter name and the associated unit type separated by a hyphen (for example, mass-kg).  The remaining rows should alternate between empty and containing raw numbers from experimental data.  Attempts to input a data file not in this form will result in error messages being printed to the console reminding the user of the correct format and pointing out the problematic data types that have been inputted.

 
