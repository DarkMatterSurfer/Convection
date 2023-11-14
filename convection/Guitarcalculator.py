import numpy as np
choice = input("Calc note or tension:")
stringlength = 0.6475 
stringmass = 0.002
notefreq = [20.60, 49.00, 32.70, 36.71]
indices = [0, 1, 2, 3]
def frequencycalc(mass, length, forcetension):
    mu = mass/length
    fund_frequency = np.sqrt(forcetension/mu)
    return fund_frequency 
def paramcalc(freq, length, mass):
    mu = mass/length
    V = 2*length*float(freq)
    paramtension = (V*V)*mu
    return paramtension
if choice == "note":
#input variables
    mass_input = float(input("Input mass in kg:"))
    length_input = float(input("Input length in m:"))
    tension_input = float(input("Input tension force in N:"))
    freq = frequencycalc(mass_input, length_input, tension_input)
    if 20.40 < freq < 20.80: 
        print("The string parameters you inputed will play a E note")
    elif 48.80 < freq < 49.20: 
        print("The string parameters you inputed will play a G note")
    elif 32.50 < freq < 32.90:
        print("The string parameters you inputed will play a C note")
    elif 36.51 < freq < 36.91:
        print("The string paramters you inputed will play a D note")
    else:
        print("String parameters won't play valid note")
else:
    for i in indices:
        f = float(notefreq[i])
        tension = paramcalc(f, stringlength, stringmass)
        print(tension)