def compute_id_switches(assignments):
    switches = 0

    for obj_id, history in assignments.items():
        prev = None
        for t in history:
            if prev is not None and t != prev:
                switches += 1
            prev = t

    return switches