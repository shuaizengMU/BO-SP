from typing import List, Tuple


from typing import List, Tuple

def get_esm_example_data(len=None) -> List[Tuple[str, str]]:
  """
  Returns example data for ESM.

  Args:
    len (int, optional): The number of examples to return. If None, returns all examples. Defaults to None.

  Returns:
    List[Tuple[str, str]]: A list of tuples containing protein names and sequences.
  """
  data = [
    ("protein1", "AKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGDDDDD"),
    ("protein2", "BALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIVVVVV"),
    ("protein3", "DKEVFDLIRDEEIRHLKALARKQISQTGMPPTRAEIAQRLGFRSPNAAEHLKALARKGVIEIVSGGGGGG"),
    ("protein4", "ERDHISQTGMPPTRAEIAQRLDHISQTGMPPTRRDEEIRHLLKEVFDLIRDEEIFDLIRDHISSWTTTTT"),
  ]
  
  if len is not None:
    data = [ (one_data[0], one_data[1][:len]) for one_data in data]
  
  return data