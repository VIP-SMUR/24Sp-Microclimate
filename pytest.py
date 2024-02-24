from uwg import UWG

# Define the .epw, .uwg paths to create an uwg object.
epw_path = "Sp24-Microclimate/resources/USA_GA_Atlanta.722190_TMY2.epw" # available in resources directory.

# path for Atlanta Airport EPW: Sp24-Microclimate/resources/USA_GA_Atlanta.722190_TMY2.epw
# path for GT Campus EPW: Sp24-Microclimate/resources/GT_33.77463936796479_-84.39704008595767_2020.epw

# Initialize the UWG model by passing parameters as arguments, or relying on defaults
model = UWG.from_param_args(epw_path=epw_path, bldheight=10, blddensity=0.5,
                            vertohor=0.8, grasscover=0.1, treecover=0.1, zone='1A')

model.generate()
model.simulate()

# Write the simulation result to a file.
model.write_epw()
