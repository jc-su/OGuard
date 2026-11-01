"""All eight TPC-H tables at a scale factor, for the 22 queries.

`generate(scale, seed)` draws region, nation, supplier, customer, part,
partsupp, orders, and lineitem with the value domains of the TPC-H
specification, clause 4.2.3, with NumPy's PCG64 from `seed`, and returns them
as pandas DataFrames with the specification's column names in its order. It
is not dbgen: its rows differ from dbgen's rows, but every column has the
specification's domain.

- Sizes: 10,000 suppliers, 200,000 parts, 800,000 partsupp rows, 150,000
  customers, and 1,500,000 orders per unit of scale, 1 to 7 lines per order,
  25 nations, and 5 regions.
- Keys: supplier, part, and customer keys run from 1. Order keys are sparse,
  the first 8 of every 32. Each part has four suppliers, `(p + i * (S / 4 +
  (p - 1) / S)) mod S + 1` for i in 0 to 3, with S suppliers and integer
  division; a line's supplier is one of its part's four, with i uniform. An
  order's customer is uniform among the customers whose key is not a multiple
  of 3, so a third of the customers have no order.
- Numbers: sizes in [1, 50], available quantities in [1, 9999], supply costs in
  [1.00, 1000.00], account balances in [-999.99, 9999.99], quantities in [1,
  50], discounts in [0.00, 0.10], and taxes in [0.00, 0.08], each uniform in
  steps of 1 or 0.01. The retail price is the specification's function of the
  part key, the extended price is the quantity times the retail price, and an
  order's total price is the sum of its lines' `extendedprice * (1 + tax) *
  (1 - discount)`, rounded to cents.
- Dates, as int32 days since 1970-01-01: order dates uniform in [1992-01-01,
  1998-12-31 - 151 days]; ship dates 1 to 121 days, commit dates 30 to 90 days
  after the order date; receipt dates 1 to 30 days after the ship date. The
  return flag is R or A at random if the receipt date is on or before the
  current date 1995-06-17 and N otherwise; the line status is O if the ship
  date is after the current date and F otherwise; the order status is F if
  all its lines are F, O if all are O, and P otherwise.
- Words: the nation and region names and keys, the part types from three
  syllable lists, the containers from two, the market segments, order
  priorities, ship instructions, and ship modes of clause 4.2.3; a part name
  is five distinct words of the 92 colors; `Manufacturer#M` and `Brand#MN`
  with M and N in [1, 5]; `Supplier#`, `Customer#`, and `Clerk#` with nine
  digits; phone numbers `CC-LLL-LLL-LLLL` whose country code is the nation key
  plus 10; addresses of 10 to 40 characters from 64 symbols.
- Comments have the specification's lengths but are not text of its grammar
  (clause 4.2.2.10): each is a substring, at a random offset, of a pool of
  random words, as dbgen takes its comments from a text pool. The pool holds
  none of the words that queries search for. `Customer` and later
  `Complaints` are written at random positions of the comments of 5 suppliers
  per unit of scale, rounded up so that every database has one (Q16), and
  `Customer` and later `Recommends` into as many others. `special` and later
  `requests` are written into 1.07 percent of the order comments, the rate in
  dbgen's data at scale factor 1, whose Q13 answer counts 1,483,918 of the
  1,500,000 orders (Q13).

The formula of the four suppliers of a part gives four distinct suppliers
only at a large enough scale; `generate` fails at a scale where it does not
(below scale factor 0.01 it can).
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.tpch.data import CURRENT, END, ORDERS_PER_SCALE, PARTS_PER_SCALE, START, day

__all__ = ["COLORS", "NATIONS", "REGIONS", "TABLES", "Tables", "generate"]

SUPPLIERS_PER_SCALE = 10_000
CUSTOMERS_PER_SCALE = 150_000
TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem")

REGIONS = ("AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST")
# each nation's name and region key, in the order of its key
NATIONS = (
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
)
COLORS = (
    "almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched",
    "blue", "blush", "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate",
    "coral", "cornflower", "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger",
    "drab", "firebrick", "floral", "forest", "frosted", "gainsboro", "ghost", "goldenrod",
    "green", "grey", "honeydew", "hot", "indian", "ivory", "khaki", "lace", "lavender",
    "lawn", "lemon", "light", "lime", "linen", "magenta", "maroon", "medium", "metallic",
    "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange", "orchid",
    "pale", "papaya", "peach", "peru", "pink", "plum", "powder", "puff", "purple", "red",
    "rose", "rosy", "royal", "saddle", "salmon", "sandy", "seashell", "sienna", "sky",
    "slate", "smoke", "snow", "spring", "steel", "tan", "thistle", "tomato", "turquoise",
    "violet", "wheat", "white", "yellow",
)  # fmt: skip
TYPE_SYLLABLES = (
    ("STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"),
    ("ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"),
    ("TIN", "NICKEL", "BRASS", "STEEL", "COPPER"),
)
CONTAINER_SYLLABLES = (
    ("SM", "LG", "MED", "JUMBO", "WRAP"),
    ("CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"),
)
SEGMENTS = ("AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD")
PRIORITIES = ("1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW")
INSTRUCTIONS = ("DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN")
MODES = ("REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB")

# the words of the comments' pool: dbgen's nouns, verbs, adjectives, adverbs,
# and prepositions, without the words that queries search for
WORDS = (
    "foxes", "ideas", "theodolites", "pinto", "beans", "instructions", "dependencies",
    "excuses", "platelets", "asymptotes", "courts", "dolphins", "multipliers", "sauternes",
    "warthogs", "frets", "dinos", "attainments", "somas", "patterns", "forges", "braids",
    "packages", "accounts", "deposits", "pearls", "tithes", "waters", "orbits", "gifts",
    "sleep", "wake", "are", "cajole", "haggle", "nag", "use", "boost", "affix", "detect",
    "integrate", "maintain", "nod", "was", "lose", "sublate", "solve", "thrash", "promise",
    "engage", "hinder", "print", "breach", "eat", "grow", "impress", "serve", "run",
    "furious", "sly", "careful", "blithe", "quick", "fluffy", "slow", "quiet", "ruthless",
    "thin", "close", "dogged", "daring", "brave", "stealthy", "permanent", "enticing",
    "idle", "busy", "regular", "final", "ironic", "even", "bold", "silent", "pending",
    "unusual", "express", "furiously", "slyly", "carefully", "blithely", "quickly",
    "fluffily", "slowly", "quietly", "ruthlessly", "boldly", "about", "above", "across",
    "after", "against", "along", "among", "around", "at", "before", "beside", "beyond",
    "by", "during", "for", "from", "into", "near", "of", "on", "over", "since", "to",
    "under", "upon", "with", "within",
)  # fmt: skip
# the words that queries search for in comments, which the pool does not hold
SEARCHED = ("special", "requests", "Customer", "Complaints", "Recommends")
POOL_WORDS = 200_000  # about 1.4 MB of text
SPECIAL_REQUESTS = 0.0107
COMPLAINTS_PER_SCALE = 5
# the 64 symbols of an address
SYMBOLS = np.frombuffer(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.", dtype=np.uint8
)
CHUNK = 1 << 20  # rows of text drawn at once


@dataclass(frozen=True)
class Tables[F]:
    """The eight tables of one database as frames of one library, and its scale factor."""

    scale: float
    region: F
    nation: F
    supplier: F
    customer: F
    part: F
    partsupp: F
    orders: F
    lineitem: F

    def frames(self) -> tuple[tuple[str, F], ...]:
        """Each table's name and frame, in the order of `TABLES`."""
        return tuple(zip(TABLES, self._frames(), strict=True))

    def _frames(self) -> tuple[F, ...]:
        return (
            self.region,
            self.nation,
            self.supplier,
            self.customer,
            self.part,
            self.partsupp,
            self.orders,
            self.lineitem,
        )

    def map[G](self, convert: Callable[[F], G]) -> "Tables[G]":
        """The same tables, each frame converted, such as to another library's frame."""
        return Tables(self.scale, *(convert(frame) for frame in self._frames()))


def _count(per_scale: int, scale: float) -> int:
    count = round(per_scale * scale)
    if count < 1:
        raise ValueError(f"scale factor {scale} gives no row of a table of {per_scale} per unit")
    return count


def _strings(values: np.ndarray) -> pd.Series:
    """A string column of NumPy strings or of a byte matrix whose rows are
    zero-padded ASCII text."""
    if values.dtype == np.uint8:
        width = values.shape[1]
        values = values.view(f"S{width}").ravel().astype(f"U{width}")
    return pd.Series(values, dtype="str")


def _words(words: tuple[str, ...], codes: np.ndarray) -> pd.Series:
    return _strings(np.asarray(words)[codes])


def _numbered(prefix: str, numbers: np.ndarray) -> pd.Series:
    """`prefix` followed by each number in nine digits, such as `Supplier#000000001`."""
    return _strings(np.strings.add(prefix, np.strings.zfill(numbers.astype("U"), 9)))


def _cents(rng: np.random.Generator, low: int, high: int, count: int) -> np.ndarray:
    """Amounts uniform in [low, high] cents, in units of 1.00."""
    return rng.integers(low, high, count, endpoint=True) / 100


def _phones(rng: np.random.Generator, nations: np.ndarray) -> pd.Series:
    """Phone numbers `CC-LLL-LLL-LLLL`, whose country code is the nation key plus 10."""
    count = len(nations)
    parts = (
        rng.integers(100, 999, count, endpoint=True),
        rng.integers(100, 999, count, endpoint=True),
        rng.integers(1000, 9999, count, endpoint=True),
    )
    text = (nations + 10).astype("U")
    for part in parts:
        text = np.strings.add(np.strings.add(text, "-"), part.astype("U"))
    return _strings(text)


def _addresses(rng: np.random.Generator, count: int) -> pd.Series:
    """Addresses of 10 to 40 characters from 64 symbols."""
    lengths = rng.integers(10, 40, count, endpoint=True)
    matrix = SYMBOLS[rng.integers(0, len(SYMBOLS), (count, 40))]
    matrix[np.arange(40) >= lengths[:, None]] = 0
    return _strings(matrix)


def _pool(rng: np.random.Generator) -> np.ndarray:
    """The comments' pool: random words, as ASCII bytes."""
    text = " ".join(np.asarray(WORDS)[rng.integers(0, len(WORDS), POOL_WORDS)].tolist())
    for word in SEARCHED:
        if word in text:
            raise ValueError(f"the comments' pool holds {word!r}, which a query searches for")
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8)


def _text(
    rng: np.random.Generator, pool: np.ndarray, count: int, low: int, high: int
) -> tuple[np.ndarray, np.ndarray]:
    """`count` substrings of the pool, of lengths uniform in [low, high], as the
    rows of a byte matrix padded with zeros, and their lengths."""
    lengths = rng.integers(low, high, count, endpoint=True)
    starts = rng.integers(0, len(pool) - high, count)
    matrix = np.empty((count, high), dtype=np.uint8)
    columns = np.arange(high)
    for begin in range(0, count, CHUNK):
        end = min(count, begin + CHUNK)
        rows = pool[starts[begin:end, None] + columns]
        rows[columns >= lengths[begin:end, None]] = 0
        matrix[begin:end] = rows
    return matrix, lengths


def _mark(
    rng: np.random.Generator,
    matrix: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    first: bytes,
    second: bytes,
) -> None:
    """Write `first` and, some characters later, `second` at random positions
    inside the text of each of `rows` of the matrix."""
    room = lengths[rows] - len(first) - len(second)
    if (room < 0).any():
        raise ValueError(f"a comment is too short for {first!r} and {second!r}")
    gap = rng.integers(0, room, endpoint=True)
    offset = rng.integers(0, room - gap, endpoint=True)
    for word, at in ((first, offset), (second, offset + len(first) + gap)):
        positions = at[:, None] + np.arange(len(word))
        matrix[rows[:, None], positions] = np.frombuffer(word, dtype=np.uint8)


def _region(rng: np.random.Generator, pool: np.ndarray) -> pd.DataFrame:
    comment, _ = _text(rng, pool, len(REGIONS), 31, 115)
    return pd.DataFrame(
        {
            "r_regionkey": np.arange(len(REGIONS), dtype=np.int64),
            "r_name": _strings(np.asarray(REGIONS)),
            "r_comment": _strings(comment),
        }
    )


def _nation(rng: np.random.Generator, pool: np.ndarray) -> pd.DataFrame:
    comment, _ = _text(rng, pool, len(NATIONS), 31, 114)
    return pd.DataFrame(
        {
            "n_nationkey": np.arange(len(NATIONS), dtype=np.int64),
            "n_name": _strings(np.asarray([name for name, _ in NATIONS])),
            "n_regionkey": np.asarray([region for _, region in NATIONS], dtype=np.int64),
            "n_comment": _strings(comment),
        }
    )


def _supplier(
    rng: np.random.Generator, pool: np.ndarray, scale: float, suppliers: int
) -> pd.DataFrame:
    keys = np.arange(1, suppliers + 1, dtype=np.int64)
    address = _addresses(rng, suppliers)
    nation = rng.integers(0, len(NATIONS) - 1, suppliers, endpoint=True)
    phone = _phones(rng, nation)
    balance = _cents(rng, -99_999, 999_999, suppliers)
    comment, lengths = _text(rng, pool, suppliers, 25, 100)
    marked = math.ceil(COMPLAINTS_PER_SCALE * scale)
    if 2 * marked > suppliers:
        raise ValueError(f"{suppliers} suppliers are too few for {marked} complaints")
    rows = rng.choice(suppliers, 2 * marked, replace=False)
    _mark(rng, comment, lengths, rows[:marked], b"Customer", b"Complaints")
    _mark(rng, comment, lengths, rows[marked:], b"Customer", b"Recommends")
    return pd.DataFrame(
        {
            "s_suppkey": keys,
            "s_name": _numbered("Supplier#", keys),
            "s_address": address,
            "s_nationkey": nation.astype(np.int64),
            "s_phone": phone,
            "s_acctbal": balance,
            "s_comment": _strings(comment),
        }
    )


def _retail_cents(partkey: np.ndarray) -> np.ndarray:
    """The retail price of each part, in cents."""
    return 90_000 + (partkey // 10) % 20_001 + 100 * (partkey % 1_000)


def _part(rng: np.random.Generator, pool: np.ndarray, parts: int) -> pd.DataFrame:
    keys = np.arange(1, parts + 1, dtype=np.int64)
    # five distinct colors in random order: the five smallest of a random key per color
    draws = rng.random((parts, len(COLORS)))
    chosen = np.argpartition(draws, 5, axis=1)[:, :5]
    chosen = np.take_along_axis(
        chosen, np.argsort(np.take_along_axis(draws, chosen, axis=1), axis=1), axis=1
    )
    colors = np.asarray(COLORS)
    name = colors[chosen[:, 0]]
    for k in range(1, 5):
        name = np.strings.add(np.strings.add(name, " "), colors[chosen[:, k]])
    maker = rng.integers(1, 5, parts, endpoint=True)
    brand = 10 * maker + rng.integers(1, 5, parts, endpoint=True)
    types = tuple(" ".join(words) for words in itertools.product(*TYPE_SYLLABLES))
    containers = tuple(" ".join(words) for words in itertools.product(*CONTAINER_SYLLABLES))
    kind = rng.integers(0, len(types), parts)
    size = rng.integers(1, 50, parts, endpoint=True)
    container = rng.integers(0, len(containers), parts)
    comment, _ = _text(rng, pool, parts, 5, 22)
    return pd.DataFrame(
        {
            "p_partkey": keys,
            "p_name": _strings(name),
            "p_mfgr": _strings(np.strings.add("Manufacturer#", maker.astype("U"))),
            "p_brand": _strings(np.strings.add("Brand#", brand.astype("U"))),
            "p_type": _words(types, kind),
            "p_size": size.astype(np.int64),
            "p_container": _words(containers, container),
            "p_retailprice": _retail_cents(keys) / 100,
            "p_comment": _strings(comment),
        }
    )


def _suppliers_of(partkey: np.ndarray, choice: np.ndarray, suppliers: int) -> np.ndarray:
    """The `choice`-th of the four suppliers of each part, `choice` in [0, 3]."""
    step = suppliers // 4 + (partkey - 1) // suppliers
    return (partkey + choice * step) % suppliers + 1


def _partsupp(
    rng: np.random.Generator, pool: np.ndarray, parts: int, suppliers: int
) -> pd.DataFrame:
    partkey = np.repeat(np.arange(1, parts + 1, dtype=np.int64), 4)
    suppkey = _suppliers_of(partkey, np.tile(np.arange(4), parts), suppliers)
    four = np.sort(suppkey.reshape(parts, 4), axis=1)
    if (four[:, 1:] == four[:, :-1]).any():
        raise ValueError(f"{suppliers} suppliers give a part the same supplier twice")
    rows = 4 * parts
    available = rng.integers(1, 9_999, rows, endpoint=True)
    cost = _cents(rng, 100, 100_000, rows)
    comment, _ = _text(rng, pool, rows, 49, 198)
    return pd.DataFrame(
        {
            "ps_partkey": partkey,
            "ps_suppkey": suppkey.astype(np.int64),
            "ps_availqty": available.astype(np.int64),
            "ps_supplycost": cost,
            "ps_comment": _strings(comment),
        }
    )


def _customer(rng: np.random.Generator, pool: np.ndarray, customers: int) -> pd.DataFrame:
    keys = np.arange(1, customers + 1, dtype=np.int64)
    address = _addresses(rng, customers)
    nation = rng.integers(0, len(NATIONS) - 1, customers, endpoint=True)
    phone = _phones(rng, nation)
    balance = _cents(rng, -99_999, 999_999, customers)
    segment = rng.integers(0, len(SEGMENTS), customers)
    comment, _ = _text(rng, pool, customers, 29, 116)
    return pd.DataFrame(
        {
            "c_custkey": keys,
            "c_name": _numbered("Customer#", keys),
            "c_address": address,
            "c_nationkey": nation.astype(np.int64),
            "c_phone": phone,
            "c_acctbal": balance,
            "c_mktsegment": _words(SEGMENTS, segment),
            "c_comment": _strings(comment),
        }
    )


def _orders_and_lineitem(
    rng: np.random.Generator,
    pool: np.ndarray,
    scale: float,
    orders: int,
    customers: int,
    parts: int,
    suppliers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = np.arange(orders, dtype=np.int64)
    orderkey = (index // 8) * 32 + index % 8 + 1
    # the k-th customer key that is not a multiple of 3
    k = rng.integers(0, customers - customers // 3, orders)
    custkey = (k // 2) * 3 + k % 2 + 1
    orderdate = rng.integers(day(START), day(END) - 151, orders, endpoint=True)
    priority = rng.integers(0, len(PRIORITIES), orders)
    clerk = rng.integers(1, _count(1_000, scale), orders, endpoint=True)
    comment, comment_lengths = _text(rng, pool, orders, 19, 78)
    special = np.flatnonzero(rng.random(orders) < SPECIAL_REQUESTS)
    _mark(rng, comment, comment_lengths, special, b"special", b"requests")

    lines = rng.integers(1, 7, orders, endpoint=True)
    owner = np.repeat(index, lines)
    rows = len(owner)
    linenumber = np.arange(rows) - np.repeat(np.cumsum(lines) - lines, lines) + 1
    partkey = rng.integers(1, parts, rows, endpoint=True)
    suppkey = _suppliers_of(partkey, rng.integers(0, 3, rows, endpoint=True), suppliers)
    quantity = rng.integers(1, 50, rows, endpoint=True)
    extendedprice = quantity * _retail_cents(partkey) / 100
    discount = _cents(rng, 0, 10, rows)
    tax = _cents(rng, 0, 8, rows)
    shipdate = orderdate[owner] + rng.integers(1, 121, rows, endpoint=True)
    commitdate = orderdate[owner] + rng.integers(30, 90, rows, endpoint=True)
    receiptdate = shipdate + rng.integers(1, 30, rows, endpoint=True)
    returned = rng.integers(0, 1, rows, endpoint=True)
    returnflag = np.where(receiptdate <= day(CURRENT), returned, 2)
    shipped = shipdate <= day(CURRENT)  # line status F
    instruction = rng.integers(0, len(INSTRUCTIONS), rows)
    mode = rng.integers(0, len(MODES), rows)
    line_comment, _ = _text(rng, pool, rows, 10, 43)

    finished = np.bincount(owner, weights=shipped, minlength=orders)
    status = np.where(finished == lines, 0, np.where(finished == 0, 1, 2))
    charge = extendedprice * (1 + tax) * (1 - discount)
    totalprice = np.round(np.bincount(owner, weights=charge, minlength=orders), 2)
    orders_frame = pd.DataFrame(
        {
            "o_orderkey": orderkey,
            "o_custkey": custkey,
            "o_orderstatus": _words(("F", "O", "P"), status),
            "o_totalprice": totalprice,
            "o_orderdate": orderdate.astype(np.int32),
            "o_orderpriority": _words(PRIORITIES, priority),
            "o_clerk": _numbered("Clerk#", clerk),
            "o_shippriority": np.zeros(orders, dtype=np.int64),
            "o_comment": _strings(comment),
        }
    )
    lineitem = pd.DataFrame(
        {
            "l_orderkey": orderkey[owner],
            "l_partkey": partkey.astype(np.int64),
            "l_suppkey": suppkey.astype(np.int64),
            "l_linenumber": linenumber.astype(np.int64),
            "l_quantity": quantity.astype(np.float64),
            "l_extendedprice": extendedprice,
            "l_discount": discount,
            "l_tax": tax,
            "l_returnflag": _words(("R", "A", "N"), returnflag),
            "l_linestatus": _words(("O", "F"), shipped.astype(np.int64)),
            "l_shipdate": shipdate.astype(np.int32),
            "l_commitdate": commitdate.astype(np.int32),
            "l_receiptdate": receiptdate.astype(np.int32),
            "l_shipinstruct": _words(INSTRUCTIONS, instruction),
            "l_shipmode": _words(MODES, mode),
            "l_comment": _strings(line_comment),
        }
    )
    return orders_frame, lineitem


def generate(scale: float, seed: int) -> Tables[pd.DataFrame]:
    """The eight tables at scale factor `scale`, drawn from `seed`."""
    rng = np.random.Generator(np.random.PCG64(seed))
    suppliers = _count(SUPPLIERS_PER_SCALE, scale)
    parts = _count(PARTS_PER_SCALE, scale)
    customers = _count(CUSTOMERS_PER_SCALE, scale)
    orders = _count(ORDERS_PER_SCALE, scale)
    pool = _pool(rng)
    region = _region(rng, pool)
    nation = _nation(rng, pool)
    supplier = _supplier(rng, pool, scale, suppliers)
    part = _part(rng, pool, parts)
    partsupp = _partsupp(rng, pool, parts, suppliers)
    customer = _customer(rng, pool, customers)
    orders_frame, lineitem = _orders_and_lineitem(
        rng, pool, scale, orders, customers, parts, suppliers
    )
    return Tables(scale, region, nation, supplier, customer, part, partsupp, orders_frame, lineitem)
