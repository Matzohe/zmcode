from src.CV.ClassHomeWork.weed_detect_test.NACLIP_extract import extract_weed_likelyhood
from src.utils.utils import INIconfig

def weed_test():
    config = INIconfig("config/seed_detect_config.cfg")
    extract_weed_likelyhood(config)