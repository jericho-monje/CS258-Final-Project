import csv

def csv_lineReader(file_path:str):
    with open(file_path, 'r') as file:
        csv_reader:csv.DictReader = csv.DictReader(file)
        for line in csv_reader:
            yield line

def _generate_req(t0):
    try:
        return next(t0)
    except StopIteration:
        return None
    
def test_csvLineReader():
    req_loader = csv_lineReader(r"C:\Users\phong\Desktop\work\CS 258\CS258-Final-Project\CS258-Final-Project\app\model\data\train\requests-0.csv")
    t1 = _generate_req(req_loader)
    while t1:
        print(str(t1))
        t1 = _generate_req(req_loader)

test_csvLineReader()