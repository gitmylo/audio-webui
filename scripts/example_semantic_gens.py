max_token = 10000  # Only change this for custom bark models with different vocab sizes


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split(lst, size):
    return list(chunks(lst, size))


def linear_full():
    return list(range(0, max_token))


def linear_split(size):
    return split(linear_full(), size)


def shuffle_full():
    import random
    _list = list(range(0, max_token))
    random.shuffle(_list)
    return _list


def shuffle_split(size):
    return split(shuffle_full(), size)


def random(count):
    import random
    return [random.randint(0, max_token-1) for i in range(count)]


def random_chunks(count, size):
    return [random(size) for i in range(count)]
