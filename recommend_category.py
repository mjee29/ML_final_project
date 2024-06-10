import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 데이터 불러오기
data = pd.read_csv('final_perfume_data.csv', encoding="latin1")

# 노트 별 대표 향 딕셔너리 예시 (실제 딕셔너리로 교체 필요)
note_categories ={'CITRUS': ['bergamot', 'bigarade', 'bitter orange', 'blood orange', 'calamansi', 'candied lemon', 'chen pi',
               'chinotto', 'citron', 'citrus water', 'citruses', 'clementine', 'finger lime', 'grapefruit',
               'grapefruit leaf', 'grapefruit peel', 'green tangerine', 'hassaku', 'hatkora lemon', 'kaffir lime',
               'kumquat', 'lemon', 'lemon balm', 'lemon tree', 'lemon verbena', 'lemon zest', 'lemongrass', 'lime',
               'limetta', 'litsea cubeba', 'mandarin orange', 'mandora', 'methyl pamplemousse', 'neroli', 'orange',
               'palestinian sweet lime', 'petitgrain', 'pokan', 'pomelo', 'rangpur', 'red mandarin', 'shiikuwasha',
               'tangelo', 'tangerine', 'tangerine zest', 'yuzu', 'green yuzu', 'italian bergamot oil grand cru','citrus'],
'FRUITS VEGETABLES NUTS': ['acai berry', 'acerola', 'acorn', 'almond', 'apple', 'apple juice', 'apple pulp', 'apple sherbet', 'apricot',
                                'arctic bramble', 'argan', 'artichoke', 'banana', 'barberry', 'bearberry', 'beetroot', 'black cherry', 'black currant',
                               'black sapote', 'black walnut', 'blackberry', 'blackthorn', 'blueberry', 'boysenberry', 'brazil nut', 'breadnut', 'buriti',
                               'burning cherry', 'candlenut', 'cantaloupe', 'carambola star fruit', 'carrot', 'cashew', 'cassowary fruit', 'cauliflower',
                               'cep', 'cepes', 'chayote', 'cherimoya', 'cherry', 'chestnut', 'chia seed', 'chickpeas', 'chinese magnolia', 'cloudberry',
                               'coco de mer', 'cocoa shell', 'coconut', 'coconut water', 'conifer', 'corn', 'corn silk', 'counts fruit', 'cranberry',
                               'cucumber', 'cupuaçu', 'currant leaf and bud', 'cyperus scariosus', 'daikon radish', 'dark plum wu mei', 'dewberry',
                               'dried apple crisp', 'dried apricot', 'dried fruits', 'durian', 'elderberry', 'feijoa fruit', 'fig', 'fig leaf',
                               'forest fruits', 'frosted berries', 'fruit salad', 'fruity notes', 'gariguette strawberry', 'genipapo', 'goji berries',
                               'gooseberry', 'grains', 'grape seed', 'grapes', 'green anjou pears', 'green grape', 'green pear', 'green plum', 'greengage',
                               'ground cherry', 'guarana', 'guava', 'hazelnut', 'hog plum', 'honeydew melon', 'isabella grape', 'jabuticaba', 'jackfruit',
                               'japanese loquat', 'jobs tears yi yi ren', 'kiwi', 'kumbaru', 'lingonberry', 'litchi', 'loganberry', 'longan berries', 'macadamia',
                               'mahonia', 'malt', 'mamey', 'mango', 'mangosteen', 'maninka', 'marian plum', 'medlar', 'melon', 'mirabelle', 'moepel accord',
                               'mulberry', 'mung bean', 'mushroom', 'nashi pear', 'nectarine', 'nutty notes', 'okra seeds', 'olive', 'papaya', 'passionfruit',
                               'pea', 'peach', 'peanut', 'pear', 'pecan', 'persimmon', 'peruvian pepper', 'pineapple', 'pinot noir grapes', 'pistachio',
                               'pitahaya', 'pitanga', 'plum', 'pomegranate', 'potato', 'prickly pear', 'pumpkin', 'quandong desert peach', 'quince', 'rambutan',
                               'raspberry', 'red apple', 'red berries', 'red currant', 'red fruits', 'red fruits smoothie', 'red mulberry', 'rhubarb', 'roasted nuts',
                               'sapodilla', 'sarsaparilla', 'sea buckthorn', 'seriguela', 'shea butter', 'shea nuts', 'silverberry', 'snowberry', 'sour cherry',
                               'soursop', 'soybean', 'squash', 'star apple', 'strawberry', 'tayberry', 'tomato', 'tropical fruits', 'tropicalone', 'tucumã', 'walnut',
                               'walnut milk', 'water fruit', 'watermelon', 'wattleseed', 'white currant', 'white grape', 'wild strawberry', 'williams pear', 'winterberry',
                               'wolfberry', 'yellow cherry', 'yellow fruits', 'yuca cassava', 'yumberry', 'lemon from italy', 'plum accord', 'italian lemon', 'bahamian orange', 'jamaican lime', 'japanese grapefruit (yuzu)', 'wildberries','blackberries', 'star fruit',
                               'kampot pepper', 'citrus fruits', 'madagascan black pepper','californian orange''japanese red chili peppers', 'candied apple', 'mediterranean blood orange', 'orange rind',
                               'roses from grasse and arabia', 'hibiscus from guinea','ivory coast cacao absolute','wild blackberry'],

'FLOWERS': ["acacia", "acerola blossom", "aglaia", "albizia", "almond blossom", "alpinia", "alstroemeria", "althaea", "alumroot", "alyssum", "amarillys", "amazon moonflower", "anemone", "angel's trumpet", "apple blossom", "apricot blossom", "ashoka flower", "astragalus", "azalea", "azteca lily",
    "banana flower", "banksia australian", "begonia", "belladona", "bellflower", "bergamot blossom", "bird cherry", "black currant blossom", "blackberry blossom", "blue lilies", "blue poppy", "bluebell", "bottlebrush", "bougainvillea", "bread flower", "bromelia", "buddleia", "butomus umbellatus", "buttercup",
    "cacao blossom", "calla lily", "camellia", "campion flower", "cananga", "cannonball flower", "carambola blossom", "cardamom flower", "carnation", "celosia", "chai hu", "chamomile", "champaca", "cherry blossom", "chimonanthus or wintersweet", "china rose", "chinotto blossom", "chocolate flower", "christmas tree or flame tree",
    "chrysanthemum", "cistus incanus", "clematis", "clover", "coconut blossom", "coffee blossom", "cornflower sultan seeds", "cosmos flower", "costus", "cotton flower", "creamy flowers", "crinum lily", "cucumber flower", "cyclamen", "dahlia", "daisy", "dandelion", "daphne", "daylily",
    "deadnettle", "delonix", "desert rose", "dianthus", "dogwood blossom", "dove tree", "dried rose", "dyer's greenweed", "edelweiss", "eglantine rose", "elderflower", "encian", "english marigold", "erigeron fleabane", "euphorbia", "eustoma | lisianthus", "evergreen", "field scabious", "fig blossom",
    "fire lily", "flamingo flower anthurium", "flax", "floral notes", "forget me not", "fragonia", "freesia", "fringed pink", "fuchsia", "geranium", "gerbera", "gladiolus", "goldenrod", "gorse", "grapeflower", "great burnet", "green nard", "green tea flower", "grevillea", "guava blossom",
    "guayacan", "gustavia flower", "hawthorn", "hazel blossom", "heather", "heliotrope", "hellabore flower", "hemlock", "hibiscus", "hoary stock", "holly flower", "hollyhock", "honeybush or cyclopia", "honeydew blossom", "hortensia", "hoya carnosa wax plant", "hyacinth", "hyssop", "impatiens",
    "inula", "iris", "iris butter", "iris pallida", "jacaranda", "jade flower", "jasmine orchid", "jujube blossom", "kadam", "kangaroo paw", "kanuka", "karmaflor", "kiwi blossom", "kudzu", "laburnum", "lady of the night flower", "lady slipper orchid", "lamduan flower", "lamprocapnos",
    "lantana", "larkspur", "laurel blossom", "lavender", "leatherwood", "ledum", "liatris", "liatrix", "lilac", "lily of the valley", "lime linden blossom", "litchi blossom", "longoza", "lotus", "lupin", "lydia broom", "lysylang", "magnolia", "magnolia brooklynensis", "magnolia leaf",
    "mahonial", "malva", "mango blossom", "mariposa lily", "mayflower", "meadowsweet", "melilotus", "melissa flower", "michelia", "mignonette", "mimosa", "mimusops elengi", "mirabilis", "monarda", "monoi oil", "morning glory flower", "moss flox", "myrtle", "narcissus", "nard",
    "nasturtium", "nectarine blossom", "nerium oleander", "nigella", "night blooming jasmine", "nom maew", "olive flower", "opium", "orange cassia tree", "orange flower water", "orchard blossom", "orchid", "orchid black diamond", "orchid cactus", "orchid pink leopard", "ornithogalum", "orris root", "osmanthus",
    "pansy", "papaya blossom", "paramela", "passion flower", "pataqueira", "peach blossom", "pear blossom", "pelargonium", "peony", "periwinkle", "petalia", "petunia", "phlox", "pineapple blossom", "pink flamingo heliconia", "pink lily", "pitahaya flower", "pittosporum", "plum blossom",
    "plumeria", "poinsettia", "pomegranate blossom", "poppy", "primrose", "princess tree paulownia", "privet", "protea", "prunella", "purple coneflower", "rangoon creeper", "raspberry blossom", "redwood flower", "reseda", "rhododendron", "rosa alba", "rosa rubiginosa", "rose", "rose hip",
    "rose japanese hamanasu", "rose mallow", "rosebay willowherb", "rosebud", "roselle", "safflower", "sainfoins", "sand lily", "sandalwood flower", "santolina", "saucer magnolia", "sea daffodil", "siberian rhododendron", "silk tree blossom", "skeleton flower diphylleia grayi", "smoketree", "snakeroot",
    "snow lotus", "snowdrops", "solomon's seal", "sophora toromiro flower", "sour cherry blossom", "spanish broom", "spiraea", "st. john's wort", "starflower", "strawberry flower", "strelitzia", "strobilanthes callosa", "sunflower", "sweet pea", "taif rose", "tamarisk", "tobacco blossom", "transparent flowers", "trillium", "tulip",
    "vanilla bahiana", "viburnum", "violet", "violet woodsorrel", "wallflower", "waratah", "water flowers", "water hyacinth", "water lily", "weeping cherry blossom",
    "white dahlia", "white ginger lily", "white lace flower", "white tea blossom", "wildflowers", "wisteria", "wrightia",
    "yellow bells", "yellow flowers", "ylang-ylang", "yuzu flower", "zinnia","lavender fine", "egyptian jasmine absolute late crop",
       'lavender bulgaria', 'patchouli indonesia', 'bullhorn orchid', 'rose absolute','and the absolutes of 15 flowers.','turkish rose',
       'carnation and violet', 'jasmine and rose absolutes.','narcissus flowers', 'osmanthus petals.',
       'bulgarian rose', 'indian tuberose', 'french lavender grand cru','egyptian jasmine', 'moroccan rose',
       'egyptian centifolia rose absolute grand cru',
       'indonesian patchouli', 'tropical flowers and spicy citruses', 'centifolia rose',
       'sicilian night blooming jasmine', 'absolute jasmine','floral',
       'grandiflorum jasmine',
       'egyptian centifolia rose absolute grand cru','roses from grasse and arabia','black currant leaves and bulgarian rose',
       'amazon magnolia',
       'notes of gardenia and white exotic florals','egyptian jasmine absolute grand cru',
       "egyptian rose absolute centifolia grand cru (also from cherifa's fields)",'cactus flower', 'patchouli from indonesia',
       'indian jasmine alcoolat \x93grand cru\x94\x9d 5%','french orris absolute grand cru', 'indian jasmine absolute','madagascan ylang-ylang',
       'ylang ylang','rose petals click here for directions ã\x97close rose delight body oil by tauer perfumes directions apply on skin',
       'rub gently until the oil is absorbed and enjoy the scent of gourmand rose petals.','gardenia (and other lovely things-- the mystery is part of the glamour)'],

'WHITE FLOWERS': ["arum lily", "belanis", "black locust", "boronia", "carissa",
    "datura", "frangipani", "gardenia", "grapefruit blossom", "honeysuckle",
    "jasmine", "karo-karounde", "lemon blossom", "lily", "mandarin orange blossom",
    "melati", "mock orange", "moon flower", "night blooming cereus", "orange blossom",
    "stephanotis", "syringa", "tangerine blossom", "tiare flower", "tuberose",
    "white flowers", "white tobacco", 'moroccan neroli oil','morrocan mimosa absolute', 'moroccan mimosa absolute'],

'GREENS HERBS FOUGERES': ["agave", "algae", "aloe vera", "ammophila beach grass", "angelica",
    "angelica root", "apple mint", "arnica", "aromatic notes", "artemisia",
    "asparagus", "assam tea", "avocado", "azolla water fern", "banana leaf",
    "barley", "barrenwort", "basil", "bay essence", "beachheather",
    "betel leaf", "bigarane", "black currant leaf", "blackberry leaf",
    "blue bugle", "borage", "borneol", "bran", "buchu or agathosma",
    "buckwheat", "buckwheat tea", "bulrush", "burdock", "cactus", "calamus",
    "calycanthus", "cangzhu", "cannabis", "caper", "capitiú", "carnation leaves",
    "catnip", "cedar leaves", "celery", "celery seeds", "centella asiatica",
    "cherry leaf", "chicory", "chive", "chuan xiong", "cilantro", "citron leaf",
    "clary sage", "coca", "coleus", "creosote bush", "crithmum", "davana",
    "deer tongue grass", "dried fallen leaves", "earl grey tea", "fermented tea",
    "fern", "flouve", "fo ti ho shou wu", "fougère accord", "fresh evergreen",
    "gajumaru banyan", "galbanum", "garlic", "genmaicha", "geranium macrorrhizum zdravetz",
    "ginkgo", "ginseng", "grape leaves", "grass", "green branches", "green forest",
    "green notes", "green pepper", "green sap", "gromwell", "guao or maiden plum",
    "hat straw", "hay", "henna", "hops", "horseweed", "immortelle", "ivy",
    "jambu", "jatamansi or spikenard", "jojoba", "juniper", "juniper berries",
    "katrafay", "katsura leaf", "khella", "kunzea", "laminaria", "lantana leaf",
    "lapsang souchong tea", "laurissilva forest", "lesser calamint", "lettuce",
    "lily-of-the-valley leaves", "limnophila aromatica", "linaloe berry", "longjing tea",
    "lovage root", "mandrake", "marigold", "marjoram", "matcha tea", "mate",
    "melilot or sweet clover", "mimosa leaves", "mint", "mistletoe", "mugwort",
    "naswar", "nettle", "nut grass", "oak leaves", "oat", "olive leaf",
    "oolong tea", "orchid leaf", "oregano", "palm leaf", "palmarosa", "pandanus",
    "parsley", "peach leaf", "pear leaf", "pesto", "petrichor", "peyote", "physcool",
    "polygonum", "portulaca or pigweed", "pu'er tea", "purslane", "red algae",
    "reed", "rice", "rooibos red tea", "roots", "rose leaf", "rose thorn",
    "rosemary", "roseroot", "rue", "rumex", "rye", "sabah snake grass", "sage",
    "sansevieria", "sap", "satureja", "saw palmetto", "seaweed", "senecio", "shiso",
    "sideritis mountain tea", "silk vine or milk broom", "skunk cabbage", "snake plant",
    "spearmint", "spinach", "stems greens", "strawberry leaf", "sugandha kokila",
    "sundew", "swartzia", "sweet grass", "tansy", "tarragon", "tea", "thistle",
    "thyme", "ti leaf cordyline", "tieguanyin tea", "tobacco", "tomato leaf",
    "torreya", "trees", "tulsi", "turnera diffusa damiana", "valerian", "vanilla leaves",
    "vine", "violet leaf", "water lily leaf", "wheat", "white meranti", "wild garlic leaf","willow-leaved", "wintergreen", "woodruff galium odoratum",
    "wormwood", "yarrow", "yunnan red tea", "pink pepper from madagascar", 'canadian cedar', 'fig tree leaves', 'wood',
    'white cedar','japanese green tea','padauk wood','lemon leaves', 'green tea','cropped wood', 'tomato leaves', 'violet leaves''mimosa ( flower',
       'leaf and stem)','fennel seed.', 'egyptian blue lotus',
       'indian blue lotus', 'hawaiian blue lotus',
       'sri lankan blue lotus', 'thai blue lotus',
       'chinese osmanthus absolute 1%',
       'chinese osmanthus\x93alcoolat\x94\x9d grand cru 10%', 'cardamom from india','venezuelan tonka absolute','tarragon from the alps',
       'jamaican amyris',
       'moroccan rosemary', 'sicilian mandarin',
       'mandarin from sicily','nanah mint'],

'SPICES': ["allspice", "anise", "asafoetida", "bay leaf", "bengal pepper", "cacao pod",
    "caraway", "cardamom", "carolina reaper", "cassia", "chutney", "cinnamon",
    "cinnamon leaf", "clove leaf", "cloves", "coffee", "coffee co2", "coffee tincture",
    "coriander", "cubeb or tailed pepper", "cumin", "curcuma turmeric", "curry",
    "curry tree", "dill", "fennel", "fenugreek", "galanga", "ginger", "green coffee",
    "guinea pepper", "indian spices", "japanese pepper", "kaempferia galanga",
    "kopi luwak coffee", "licorice", "mace", "mustard seed", "nutmeg", "oily notes",
    "oriental notes", "pepper", "peppertree", "pimento", "pimento leaf", "pimento seeds",
    "pink pepper", "priprioca", "saffron", "safraleine", "sesame", "siam cardamom",
    "sichuan pepper", "spicy notes", "star anise", "sumac", "tamarind", "timur",
    "tonka bean", "toscanol", "ultravanil", "vanilla", "wan sao lhong", "wasabi", "water pepper", "west indian bay", 'chinese ginger',
    "vanilla bean", "cinnamon from madagascar", "tobacco and coffee", "madagascan vanilla olã©orã©sin", 'subtle spice notes',
    'subtle spice notes','vanilla pod', 'madagascar vanilla','hints of island vanilla orchid recent press voted top 5 cult fragrance by style.com. wallpaper magazine ym elle vogueus weeklyin touch',
    'hints of island vanilla orchid', 'tahitian vanilla','vanilla (and secret ingredients that we are not allowed to divulge)',
    'vanilla from comoros islands','madagascan vanilla \x93grand cru\x94\x9d 2.5%','tonak bean', 'clove','spicy'
    ],

'SWEETS GOURMAND': ["acetyl furan", "agave nectar", "apple pie", "baba italian dessert", "baked apple",
    "baklava", "biscuit", "bonbon", "bread", "brioche", "brown sugar", "brownie", "bubbaloo",
    "bubble gum", "burnt sugar", "butter", "buttercream", "butterscotch", "cacao butter",
    "cake", "calissons d'aix", "candied fruits", "candied ginger", "candied orange", "candies",
    "canelé", "caramel", "cassata siciliana", "cereals", "cheesecake", "cherry milk",
    "cherry syrup", "chocolate fudge", "chocolate truffle", "churros", "coconut pie",
    "coconut powder", "condensed milk", "cone waffle", "confetti sugared almonds", "cookie",
    "cookie dough", "cosmofruit iff", "cotton candy", "cream", "creamsicle", "crème brûlée",
    "croissant", "cupcake", "custard", "dark chocolate", "dates", "donut or doughnut", "dulce de leche",
    "eggnog", "fougassette", "french pastries", "frosting glacé", "gelatin", "gianduia", "gingerbread",
    "gourmand accord", "griotte cherries", "halva", "hazelnut cocoa spread", "honey", "honeycomb",
    "horchata", "ice cream", "icing pink", "jelly", "jellybean", "kulfi", "lemon pie", "loukhoum",
    "macarons", "maple syrup", "maraschino cherry", "marmalade", "marron glacé", "marshmallow",
    "marzipan", "meringues", "milk cream", "milk mousse", "milkshake", "molasses", "nougat",
    "nutella", "oatmilk", "palm sugar", "pancake", "panettone", "panna cotta", "pastiera napoletana",
    "peach cream", "pear ice cream", "popcorn", "praline", "profiterole", "pudding", "puff pastry",
    "rainbow sorbet", "red fruits sorbet", "rice pudding", "rose jam", "sacher torte", "sorbet",
    "souffle", "sprinkles", "spun sugar", "strawberry syrup", "sugar", "sugar syrup", "tartine",
    "tiramisu", "toast", "toffee", "tropézienne tarte", "vanilla caviar", "waffle", "white chocolate",
    "white chocolate truffle", "whoopie pie", "yogurt", "zefir",'strawberry salt','sweet'],

'WOODS MOSSES': ["agarwood oud", "akigalawood", "alder", "almond tree", "amaranth", "amburana bark",
    "amburana wood", "amyris", "apple tree", "araucaria", "arbutus madrona, bearberry tree",
    "argan tree", "aspen", "australian blue cypress", "bamboo", "baobab", "bark", "beech",
    "belambra tree", "birch", "black hemlock or tsuga", "black spruce", "blackwood",
    "blonde woods", "brazilian rosewood", "buddha wood", "buxus", "cabreuva", "cambodian oud",
    "canadian balsam", "carob tree", "cascarilla", "cashmir wood", "cedar", "chalood bark",
    "cherry tree", "chinese oud", "chypre notes", "clearwood", "cocobolo", "coconut tree",
    "coffee tree", "cork", "cottonwood poplar", "cypress", "cypriol oil or nagarmotha",
    "dark patchouli", "ditax wood", "dreamwood", "driftwood", "dry wood", "ducke", "ebony tree",
    "elm", "eucalyptus", "fig tree", "fir", "grass tree", "guaiac wood", "hiba", "hinoki wood",
    "ho wood", "incienso", "indian oud", "indian sandalwood", "indian woods", "ironwood",
    "ishpink, ocotea quixos", "kowhai", "laotian oud", "larch", "lichen", "liquidambar",
    "mahogany", "malaysian oud", "mango tree", "manuka", "maple", "massoia", "mesquite wood",
    "muhuhu", "mulberry plant", "mysore sandalwood", "neem", "nootka", "oak", "oakmoss",
    "olive tree", "palisander rosewood", "palo santo", "palo verde tree", "pamplewood",
    "paperbark", "papyrus", "patchouli", "patchouli green", "peach tree", "pear tree",
    "pepperwood or hercules club", "pine tree", "pink ipê tree", "plum tree", "pua keni keni pua-lulu",
    "ravenala", "ravensara", "red willow", "redwood", "saman", "sandalwood", "satinwood", "sawdust",
    "scots pine variant", "selaginella tamariscina", "sequoia", "siam", "siam wood", "spruce",
    "sycamore", "takamaka", "tamboti wood", "tatami", "teak wood", "thailand oud", "thanaka wood",
    "thuja", "transparent woods", "vetiver", "vietnamese oud", "white oud", "white willow",
    "wood barrel", "woods", "yohimbe", "rosewood", "agarwood", 'indonesian patchouli oil','chinese cedarwood',
       'brazilian palisander wood', 'woods and moss', '100% pure oud', 'aloe wood''exotic sandalwoods. click here for directions ã\x97close bay rum by gilbert henry directions natural contents will settle.  shake gently before using.'
       ,"grey amber absolute (we're pretty sure there's more to it)",
       'indian sandalwood grand cru','cedarwood','haitian vetiver\x93grand cru\x94\x9d 10%','alpine spruce oil', 'pure cambodian oud',
       ],

'RESINS BALSAMS': ["amberwood", "andiroba", "bakhoor", "balsamic notes", "balsamic vinegar",
    "benzoin", "birch tar", "bisabolene", "breu-branco", "bushman candle",
    "cade oil", "choya loban", "choya nakh", "choya ral", "coal tar",
    "copahu balm", "copaiba balm", "copal", "dragon blood resin", "elemi",
    "estoraque", "gurjun balsam", "incense", "labdanum", "mastic or lentisque",
    "mopane", "myrica", "myrrh", "nag champa", "olibanum frankincense","frankincense",
    "olibanum sacra resin green", "opoponax", "peru balsam", "pine tar",
    "poplar populus buds", "resins", "rubber", "styrax", "surf wax", "tea tree oil", 'somalian olibanum',
    'madagascan clove bud oil','balsam', 'peruvian balsam',
       'balsam tolu', 'sicilian tarocco orange oil','haitian vetiver oil special grand cru','somalian myrrh 15%'],

'MUSK AMBER ANIMALIC': ["akashic acord", "aldron", "amber", "amber xtreme", "ambergris",
    "ambertonic iff", "ambrarome", "ambrette musk mallow",
    "ambrettolide", "ambrocenide symrise", "ambrostar", "ambroxan",
    "animal notes", "bacon", "bbq", "beeswax", "carrot seeds", "castoreum",
    "caviar", "cetalox", "cheese", "civet", "civettone", "coral reef",
    "daim", "exaltolide", "fur", "genet", "goat hair", "goat's milk",
    "habanolide", "hyraceum", "kephalis", "kyphi", "leather", "meat",
    "milk", "musks", "muskrat", "oysters", "saffiano leather", "sea shells",
    "skatole", "skin", "starfish", "suede", "sylkolide", "tolu balsam",
    "truffle", "velvione",'white musks', 'musks "from the pheromone family"', 'sky aldehydes','musky amber', 'benzoin from laos',
    'sparkling musk', 'oceany', '11 select (but undisclosed!) notes that evoke vegetable tanned horse leather','musk'],

'BEVERAGES': ["absinthe", "advocaat", "amaretto", "applejack", "baileys irish cream",
    "batida", "beer", "beer ale", "bellini", "blue margarita",
    "boozy notes", "bourbon whiskey", "cachaça", "caipirinha", "campari",
    "cappuccino", "champagne", "champagne rosé", "cherry liqueur",
    "chinotto", "coca-cola", "cocktail fruits", "cognac",
    "cosmopolitan cocktail", "cream soda", "curaçao", "daiquiri",
    "eau de vie", "espresso", "fruit tea", "gin", "grenadine", "hi-fi",
    "ice wine", "jasmine tea", "kava drink", "kir royal", "lemon soda",
    "lemonade", "limoncello", "liquor", "macchiato", "madeira",
    "mai tai cocktail", "margarita", "martini", "masala chai", "mezcal",
    "midori", "mocha", "mojito", "moscow mule", "mulled wine",
    "orange soda", "ouzo", "pina colada", "pisco sour cocktail", "plum wine",
    "port wine", "prosecco", "punch", "raki", "red wine", "rhum agricole",
    "rice water", "rum", "sake", "sangria", "sparkling water",
    "sparkling wine", "syrup", "tequila", "tokaji wine", "tonic water",
    "umeshu", "vermouth", "vinegar", "vodka", "whiskey", "white wine", "wine lees", "wine must"],

'NATURAL SYNTHETIC POPULAR WEIRD': ["pepperwood", "accord eudora", "alcantara accord", "aldambre",
    "aldehydes", "aluminum", "ambreine", "ambrinol", "ambrofix",
    "ambrox super", "ammonia", "amyl salicylate", "antillone", "aqual",
    "aquozone", "ash", "asphalt", "azarbre", "black diamond",
    "black leather", "blood", "boisiris", "bourgeonal", "brick",
    "brown scotch tape", "burnt match", "calone", "calypsone", "camphor",
    "candle wax", "canvas", "caoutchouc", "cascalone", "cashalox",
    "cashmeran", "cetonal", "chalk", "cinnamaldehyde", "clarycet", "clay",
    "co2 extracts", "coal", "cobblestone", "cocaine", "concrete", "copper",
    "coral limestone", "coranol", "cork", "cosmone", "coumarin",
    "credit cards", "crustaceans", "cyclopidene", "damascone", "dew drop",
    "dihydromyrcenol", "dirt", "dodecanal", "dust", "earth tincture",
    "earthy notes", "egg", "ember", "ethyl maltol", "eugenol", "evernyl",
    "fabric", "factor x", "fire", "fish", "flint", "floralozone", "flour",
    "galaxolide", "gasoline", "georgywood", "geosmin", "gold", "graphite",
    "guaiacol", "gunpowder", "hair pomade", "hand cream",
    "head space waterfall", "healingwood", "hedione", "helvetolide",
    "hexenyl green", "hexyl acetate", "hina", "hivernal", "holy water",
    "hot iron", "ice", "indole", "industrial glue", "ink", "instant film accord",
    "iodine", "ionones", "iso e super", "isobutyl quinoline", "jasmolactone",
    "jasmone", "javanol", "jeans", "lacquered wood", "lactones", "latex",
    "lava", "lilybelle", "linen", "lip gloss", "lipstick", "little doll strawberry",
    "lorenox", "magnolan", "mascarpone cheese", "melonal", "metallic notes",
    "mineral notes", "mitti attar", "molasses", "money", "motor oil",
    "mountain air", "mousse de saxe", "mud", "mugane", "muscenone", "mystikal",
    "nail polish", "naturalcalm", "neoprene", "norlimbanol", "nympheal",
    "old books", "old house", "orcanox™", "ozonic notes", "paper",
    "para-cresyl phenyl acetate", "paradisone", "parchment", "pearadise",
    "pearls", "peat", "pebbles", "pencil", "petroleum", "pharaone",
    "pink crystal", "pink himalayan sea salt", "pizza", "plastic",
    "plastic bag", "play-doh", "poison", "poivrol", "pollen", "pomarose",
    "powdery notes", "priest’s clothes", "propolis", "prunol", "rain notes",
    "re base", "rhodinol", "rice powder", "rose oxide", "salicylic acid",
    "salt", "sand", "satin", "sauna", "scent trek", "sclarene", "sea water",
    "serenolide", "shamama attar", "silk", "silver", "sinfonide", "siren",
    "slate", "smoke", "snow", "soap", "sodium silicate", "solar notes",
    "sp3 carbon", "spiranol", "spray paint", "steam accord", "stone",
    "sulphur", "suntan lotion", "sweat", "t-shirt accord", "talc", "tar",
    "tennis ball", "terpineol", "terranol", "timbersilk", "tomato sauce",
    "tonalide", "tonquitone", "toothpaste", "trimofix", "tuberolide",
    "tulle accord", "vanillin", "varnish accord", "velvet", "verdox", "vinyl",
    "vinyl guaiacol", "vitamin c", "water", "wet plaster", "wet stone",
    "white leather", "wool", "yeast", 'ambergris infusion','silver ambergris', 'light musc','fossilized amber','galaxolide super',
    'african stone absolute','essence of distilled scotch'],

'UNCATEGORIZED': ['soy milk', 'taro','nan','and after that', 'your guess is as good as ours.']
}

# Notes 열 전처리
data = data.dropna(subset=['Notes'])
data['Processed_Notes'] = data['Notes'].apply(
    lambda x: [note.strip().lower() for note in x.split(',')] if isinstance(x, str) else []
)

# 카테고리 매핑
def map_to_category(notes, note_categories):
    categories = set()
    for note in notes:
        for category, ingredients in note_categories.items():
            if note in ingredients:
                categories.add(category)
    return ','.join(categories) if categories else 'Unknown'

# Example note categories dictionary
note_categories = {
    'MUSK AMBER ANIMALIC': ['musk', 'amber', 'animalic'],
    'SPICES': ['pepper', 'cinnamon', 'clove'],
    # Add more categories and notes as needed
}

data['category'] = data['Processed_Notes'].apply(lambda x: map_to_category(x, note_categories))

# 분석에 필요없는 컬럼 제거
data.drop(['Brand', 'Image URL', 'Description'], axis=1, inplace=True)

# Word2Vec 모델 학습
model = Word2Vec(data['category'].apply(lambda x: x.split(',')), vector_size=100, window=5, min_count=1, workers=4)

# 카테고리 벡터화 함수 정의
def vectorize_category(category_list):
    vectors = [model.wv[category] for category in category_list if category in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 데이터프레임에 벡터 추가
data['Vector'] = data['category'].apply(lambda x: vectorize_category(x.split(',')))

# 코사인 유사도 계산 함수
def calculate_similarity(perfume_vector, user_vector):
    return cosine_similarity([perfume_vector], [user_vector])[0][0]

# 향수 추천 함수
def recommend_perfumes(user_vector, top_n=5):
    perfumes_with_similarity = []
    for _, row in data.iterrows():
        similarity = calculate_similarity(row["Vector"], user_vector)
        perfumes_with_similarity.append({"Name": row["Name"], "similarity": similarity})
    # 유사도가 높은 순서대로 정렬
    recommended_perfumes = sorted(perfumes_with_similarity, key=lambda x: x["similarity"], reverse=True)
    return recommended_perfumes[:top_n]

# 추천 실행 함수
def run_recommendation():
    st.header("Recommend by Category")

    categories = [
        'MUSK AMBER ANIMALIC', 'SPICES', 'CITRUS', 'FLOWERS', 'WHITE FLOWERS',
        'RESINS BALSAMS', 'WOODS MOSSES', 'GREENS HERBS FOUGERES', 'SWEETS GOURMAND',
        'FRUITS VEGETABLES NUTS', 'BEVERAGES', 'NATURAL SYNTHETIC', 'POPULAR', 'WEIRD'
    ]

    # 줄바꿈을 추가하는 함수
    def add_line_breaks(text):
        return text.replace(',', ',<br>')

    # 설명 텍스트
    description = ('top - 가장 빨리 날아가는 향, '
                   'middle - 중간 지속 향, '
                   'base - 가장 오래 지속되는 향 / '
                   '해당 기준에 맞춰 원하는 향을 입력해주세요')

    # 줄바꿈이 추가된 설명 텍스트
    formatted_description = add_line_breaks(description)

    # Streamlit에 출력 (Markdown 형식)
    st.markdown(formatted_description, unsafe_allow_html=True)

    top_category = st.selectbox("Select the top:", categories)
    middle_category = st.selectbox("Select the middle:", categories)
    base_category = st.selectbox("Select the base:", categories)

    if st.button("Recommend by Category"):
        user_input = [top_category, middle_category, base_category]

        # 벡터 변환
        user_vector_top = vectorize_category([top_category])
        user_vector_middle = vectorize_category([middle_category])
        user_vector_base = vectorize_category([base_category])

        # 가중치 설정
        weights = {"top": 0.2, "middle": 0.3, "base": 0.5}

        # 사용자 벡터 계산
        user_vector = (weights["top"] * user_vector_top +
                       weights["middle"] * user_vector_middle +
                       weights["base"] * user_vector_base)

        # 향수 추천 함수
        recommended_perfumes = recommend_perfumes(user_vector, top_n=5)

        # DataFrame으로 추천 결과 변환
        recommended_df = pd.DataFrame(recommended_perfumes)

        st.write("\nTop 5 recommended perfumes by category:")
        st.dataframe(recommended_df)

# Streamlit 실행
if __name__ == "__main__":
    run_recommendation()
