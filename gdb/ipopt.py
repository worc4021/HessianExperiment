import gdb
import re
import numpy as np

class IpoptDenseVectorPrinter:
    def __init__(self, val: gdb.Value):
        dim = val['owner_space_']['dim_']
        values = [float(val['values_'][i]) for i in range(dim)]
        self.val = np.array(values)

    def to_string(self):
        return str(self.val)

    def display_hint(self) -> str:
        return "Ipopt::DenseVector"

def ipopt_lookup(val : gdb.Value):
    if re.search(r'\bIpopt::DenseVector\b', str(val.type.strip_typedefs())):
        return IpoptDenseVectorPrinter(val)



import sys
print(sys.version)
print("Registering Ipopt pretty printers.")
gdb.pretty_printers.append(ipopt_lookup)
print("Registered Ipopt pretty printers. Version %LOCALBUILD%")