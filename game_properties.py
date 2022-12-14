import numpy as np
from itertools import chain

CITIES_LIST = ['algiers', 'atlanta', 'baghdad', 'bangkok', 
			'beijing', 'bogota', 'buenos_aires', 'cairo', 'chennai', 'chicago', 
			'delhi', 'essen', 'ho_chi_minh_city', 'hong_kong', 'istanbul', 'jakarta', 
			'johannesburg', 'karachi', 'khartoum', 'kinshasa', 'kolkata', 'lagos', 
			'lima', 'london', 'los_angeles', 'madrid', 'manila', 'mexico_city', 
			'miami', 'milan', 'montreal', 'moscow', 'mumbai', 'new_york', 'osaka', 
			'paris', 'riyadh', 'san_francisco', 'santiago', 'sao_paulo', 'seoul', 
			'shanghai', 'st_petersburg', 'sydney', 'taipei', 'tehran', 'tokyo', 
			'washington']

NO_CITIES = 48
NO_COLOURS = 4

CITIES_EYE = [list(x) for x in np.eye(NO_CITIES)]
PLAYERROLE_EYE = [list(x) for x in np.eye(4)]
INT3_EYE = [list(x) for x in np.eye(3)]
INT4_EYE = [list(x) for x in np.eye(4)]

def int_to_bin_list(x, max_x):
	return [int(i<x) for i in range(max_x)]

ACTIONS_LIST = [\
	[0] * NO_CITIES, #DISCARD
	#Self Movement 
	[0] * NO_CITIES, #DRIVE_FERRY, normal move
	[0] * NO_CITIES, #DIRECT_FLIGHT, discard new location move
	[0] * NO_CITIES, #CHARTER_FLIGHT, discard current location move
	[0] * NO_CITIES, #SHUTTLE_FLIGHT, research station move
	#Special Actions
	[0] * NO_CITIES, #GIVE_KNOWLEDGE
	[0] * NO_CITIES, #RECEIVE_KNOWLEDGE
	[0] * (NO_CITIES + 1), #BUILD_RESEARCHSTATION, replace research station if required
	[0] * NO_COLOURS, #TREAT_DISEASE
	[0] * NO_COLOURS, #DISCOVER_CURE
]
ACTIONS_LIST = list(chain.from_iterable(ACTIONS_LIST))

NO_CITY_SELECTIONS = 8
CITY_SELECTION_OFFSET = NO_CITIES * NO_CITY_SELECTIONS
	
#Taken from https://github.com/BlopaSc/PAIndemic
city_cards = {
	'algiers': {'country':'algeria','color':'black','pop':2946000,'pop_density':6500,'connects':['madrid', 'paris', 'istanbul', 'cairo']},
	'atlanta': {'country':'united_states','color':'blue','pop':4715000,'pop_density':700,'connects':['chicago', 'washington', 'miami']},
	'baghdad': {'country':'iraq','color':'black','pop':6204000,'pop_density':10400,'connects':['istanbul', 'cairo', 'tehran', 'karachi', 'riyadh']},
	'bangkok': {'country':'thailand','color':'red','pop':7151000,'pop_density':3200,'connects':['chennai', 'kolkata', 'hong_kong', 'ho_chi_minh_city', 'jakarta']},
	'beijing': {'country':'peoples_republic_of_china','color':'red','pop':17311000,'pop_density':5000,'connects':['seoul', 'shanghai']},
	'bogota': {'country':'colombia','color':'yellow','pop':8702000,'pop_density':21000,'connects':['mexico_city', 'miami', 'sao_paulo', 'buenos_aires', 'lima']},
	'buenos_aires': {'country':'argentina','color':'yellow','pop':13639000,'pop_density':5200,'connects':['bogota', 'sao_paulo']},
	'cairo': {'country':'egypt','color':'black','pop':14718000,'pop_density':8900,'connects':['khartoum', 'algiers', 'istanbul', 'baghdad', 'riyadh']},
	'chennai': {'country':'india','color':'black','pop':8865000,'pop_density':14600,'connects':['mumbai', 'delhi', 'kolkata', 'bangkok', 'jakarta']},
	'chicago': {'country':'united_states','color':'blue','pop':9121000,'pop_density':1300,'connects':['san_francisco', 'montreal', 'atlanta', 'mexico_city', 'los_angeles']},
	'delhi': {'country':'india','color':'black','pop':22242000,'pop_density':11500,'connects':['tehran', 'karachi', 'mumbai', 'kolkata', 'chennai']},
	'essen': {'country':'germany','color':'blue','pop':575000,'pop_density':2800,'connects':['london', 'paris', 'st_petersburg', 'milan']},
	'ho_chi_minh_city': {'country':'vietnam','color':'red','pop':8314000,'pop_density':9900,'connects':['bangkok', 'jakarta', 'hong_kong', 'manila']},
	'hong_kong': {'country':'peoples_republic_of_china','color':'red','pop':7106000,'pop_density':25900,'connects':['kolkata', 'bangkok', 'shanghai', 'taipei', 'manila', 'ho_chi_minh_city']},
	'istanbul': {'country':'turkey','color':'black','pop':13576000,'pop_density':9700,'connects':['milan', 'st_petersburg', 'algiers', 'moscow', 'baghdad', 'cairo']},
	'jakarta': {'country':'indonesia','color':'red','pop':26063000,'pop_density':9400,'connects':['chennai', 'bangkok', 'ho_chi_minh_city', 'sydney']},
	'johannesburg': {'country':'south_africa','color':'yellow','pop':3888000,'pop_density':2400,'connects':['kinshasa', 'khartoum']},
	'karachi': {'country':'pakistan','color':'black','pop':20711000,'pop_density':25800,'connects':['baghdad', 'riyadh', 'tehran', 'delhi', 'mumbai']},
	'khartoum': {'country':'sudan','color':'yellow','pop':4887000,'pop_density':4500,'connects':['lagos', 'kinshasa', 'johannesburg', 'cairo']},
	'kinshasa': {'country':'democratic_republic_of_the_congo','color':'yellow','pop':9046000,'pop_density':15500,'connects':['lagos', 'khartoum', 'johannesburg']},
	'kolkata': {'country':'india','color':'black','pop':14374000,'pop_density':11900,'connects':['delhi', 'chennai', 'hong_kong', 'bangkok']},
	'lagos': {'country':'nigeria','color':'yellow','pop':11547000,'pop_density':12700,'connects':['sao_paulo', 'khartoum', 'kinshasa']},
	'lima': {'country':'peru','color':'yellow','pop':9121000,'pop_density':14100,'connects':['mexico_city', 'bogota', 'santiago']},
	'london': {'country':'united_kingdom','color':'blue','pop':8568000,'pop_density':5300,'connects':['new_york', 'essen', 'paris', 'madrid']},
	'los_angeles': {'country':'united_states','color':'yellow','pop':14900000,'pop_density':2400,'connects':['san_francisco', 'chicago', 'mexico_city', 'sydney']},
	'madrid': {'country':'spain','color':'blue','pop':5427000,'pop_density':5700,'connects':['new_york', 'london', 'paris', 'algiers', 'sao_paulo']},
	'manila': {'country':'philippines','color':'red','pop':20767000,'pop_density':14400,'connects':['san_francisco', 'hong_kong', 'ho_chi_minh_city', 'taipei', 'sydney']},
	'mexico_city': {'country':'mexico','color':'yellow','pop':19463000,'pop_density':9500,'connects':['chicago', 'los_angeles', 'miami', 'bogota', 'lima']},
	'miami': {'country':'united_states','color':'yellow','pop':5582000,'pop_density':1700,'connects':['atlanta', 'mexico_city', 'washington', 'bogota']},
	'milan': {'country':'italy','color':'blue','pop':5232000,'pop_density':2800,'connects':['paris', 'essen', 'istanbul']},
	'montreal': {'country':'canada','color':'blue','pop':3429000,'pop_density':2200,'connects':['chicago', 'new_york', 'washington']},
	'moscow': {'country':'russia','color':'black','pop':15512000,'pop_density':3500,'connects':['st_petersburg', 'istanbul', 'tehran']},
	'mumbai': {'country':'india','color':'black','pop':16910000,'pop_density':30900,'connects':['karachi', 'delhi', 'chennai']},
	'new_york': {'country':'united_states','color':'blue','pop':20464000,'pop_density':1800,'connects':['montreal', 'london', 'madrid', 'washington']},
	'osaka': {'country':'japan','color':'red','pop':2871000,'pop_density':13000,'connects':['taipei', 'tokyo']},
	'paris': {'country':'france','color':'blue','pop':10755000,'pop_density':3800,'connects':['london', 'madrid', 'essen', 'milan', 'algiers']},
	'riyadh': {'country':'saudi_arabia','color':'black','pop':5037000,'pop_density':3400,'connects':['cairo', 'baghdad', 'karachi']},
	'san_francisco': {'country':'united_states','color':'blue','pop':5864000,'pop_density':2100,'connects':['chicago', 'los_angeles', 'manila', 'tokyo']},
	'santiago': {'country':'chile','color':'yellow','pop':6015000,'pop_density':6500,'connects':['lima']},
	'sao_paulo': {'country':'brazil','color':'yellow','pop':20186000,'pop_density':6400,'connects':['madrid', 'bogota', 'lagos', 'buenos_aires']},
	'seoul': {'country':'south_korea','color':'red','pop':22547000,'pop_density':10400,'connects':['beijing', 'shanghai', 'tokyo']},
	'shanghai': {'country':'peoples_republic_of_china','color':'red','pop':13482000,'pop_density':2200,'connects':['beijing', 'seoul', 'tokyo', 'hong_kong', 'taipei']},
	'st_petersburg': {'country':'russia','color':'blue','pop':4879000,'pop_density':4100,'connects':['essen', 'moscow', 'istanbul']},
	'sydney': {'country':'australia','color':'red','pop':3785000,'pop_density':2100,'connects':['los_angeles', 'jakarta', 'manila']},
	'taipei': {'country':'taiwan','color':'red','pop':8338000,'pop_density':7300,'connects':['hong_kong', 'osaka', 'manila', 'shanghai']},
	'tehran': {'country':'iran','color':'black','pop':7419000,'pop_density':9500,'connects':['moscow', 'baghdad', 'delhi', 'karachi']},
	'tokyo': {'country':'japan','color':'red','pop':13189000,'pop_density':6030,'connects':['san_francisco', 'shanghai', 'seoul', 'osaka']},
	'washington': {'country':'united_states','color':'blue','pop':4679000,'pop_density':1400,'connects':['montreal', 'new_york', 'atlanta', 'miami']},
}

if __name__=="__main__":
	print(INT3_EYE)
	
