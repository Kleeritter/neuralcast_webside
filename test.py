my_string = "abc_def_ghi_jkl_mno"
parts = my_string.split('_')
if len(parts) >= 3:
    result = '_'.join(parts[:3])  # Hier werden die ersten drei Teile genommen, inklusive dem Trennzeichen _
    print(result)
else:
    print("String enthält nicht genügend Trennzeichen.")
