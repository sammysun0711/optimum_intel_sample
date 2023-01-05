import cpuinfo

def enable_overflow_fix():
    flags = cpuinfo.get_cpu_info()['flags']
    brand_raw = cpuinfo.get_cpu_info()['brand_raw']
    w = "without"
    overflow_fix = 'enable'
    for flag in flags:
        if 'vnni' or 'amx_int8' in flag:
            w = "with"
            overflow_fix = 'disable'
    print("Detected CPU platform {} {} Intel(R) Deep Learning Boost (VNNI) technology and further generations, {}d overflow fix".format(brand_raw, w, overflow_fix))

    return overflow_fix
