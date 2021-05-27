import numpy as np 

class Config(object):
    def __init__(self):
        self.eval_windowing = False
        self.data_smoothing = True
        self.rawdata_dir = '../data/xlsx/pool/'
        self.npy_dest_dir = '../data/npy/'
        self.features = [ 
            'heat_flux',
            'temperature',
            'force',
            'material', 
            'roughness',
        ]
        # self.scaler = (
        #     0.002, #hf
        #     40, # temp
        #     800, # force
        #     10, # material
        #     50, # roughness
        # )
        self.scaler = ( #OFF
            1,
            1,
            1, 
            1,
            1,
        ) 
        self.material_str_list = [
            'Balsa',
            'MDF-thick',
            'ABS',
            'Bakelite',
            'Al',
            'Acryl',
            'ABS',
            'Nylon',
            'Cu',
            'Brass',
            'Foamex',
            'PC',
            'Stainless',
            'Walnut',
            'EcoFlex',
            'PDMS',
            'RedOak'
        ]
        self.num_features = len(self.features)
        self.output_space = []
        self.materials = {
            'Balsa': 0,
            'MDF-thick': 1,
            'ABS': 2,
            'Bakelite': 3,
            'Al': 4,
            'Acryl': 5,
            'ABS': 6,
            'Nylon': 7,
            'Cu': 8,
            'Brass': 9,
            'Foamex': 10,
            'PC': 11,
            'Stainless': 12,
            'Walnut': 13,
            'EcoFlex': 14,
            'PDMS': 15,
            'RedOak': 16
        }
        self.Ra_dict = {
            'Balsa' :{
                '40': 19.69,
                '100': 13.5,
                '400': 10.32,
                '8000': 7.69
            },
            'MDF-thick' :{
                '40': 20.84,
                '100': 11.28,
                '400': 4.18,
                '8000': 4.31
            },
            'ABS' :{
                '40': 14.3,
                '100': 4.39,
                '400': 3.14,
                '8000': 2.14
            },
            'Bakelite' :{
                '40': 6.96,
                '100': 3.34,
                '400': 2.5,
                '8000': 1.5
            },
            'Al' :{
                '40': 3.66,
                '100': 2.59,
                '400': 2.13,
                '7000': 1.41
            },
            'Acryl':{
                '40': 3.92,
                '100': 2.53,
                '400': 2.31,
                '8000': 0.93
            },
            'ABS':{
                '40': 14.3,
                '100': 4.39,
                '400': 3.14,
                '8000': 2.14
            },
            'Nylon':{
                '40': 8.14,
                '100': 8.81,
                '400': 2.63,
                '8000': 1.73
            },
            'Cu':{ #Need to be corrected
                '40': 3.66,
                '100': 2.59,
                '400': 2.13,
                '8000': 1.41
            },
            'Brass':{ #Need to be corrected
                '40': 3.66,
                '100': 2.59,
                '400': 2.13,
                '8000': 1.41
            },
            'Foamex':{
                '40': 17.8,
                '100': 7.3,
                '400': 2.58,
                '8000': 2.8
            },
            'PC':{ 
                '40': 9.55,
                '100': 4.06,
                '400': 1.87,
                '8000': 1.31
            },
            'Stainless':{ #Need to be corrected
                '40': 3.66,
                '100': 2.59,
                '400': 2.13,
                '8000': 1.41
            },
            'Walnut':{ 
                '40': 14.11,
                '100': 9.6,
                '400': 4.02,
                '8000': 2.6
            },
            'EcoFlex':{ #Need to be corrected
                '40': 17.8,
                '100': 7.3,
                '400': 2.58,
                '8000': 2.8
            },
            'PDMS':{ #Need to be corrected
                '40': 17.8,
                '100': 7.3,
                '400': 2.58,
                '8000': 2.8
            },
            'RedOak':{ 
                '40': 32.12,
                '100': 6.3,
                '400': 2.81,
                '8000': 3.02
            }
        }
        self.ep = 500
        self.batch_size = 16
        self.n_steps = 10


config = Config()
