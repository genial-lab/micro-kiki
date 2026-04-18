#!/usr/bin/env python3
"""Generate Lua / MicroPython training dataset.

Covers: Lua basics, MicroPython (machine module, GPIO, I2C, SPI, UART),
ESP32 MicroPython, Lua in embedded, LOVE2D, Neovim Lua, MicroPython vs CircuitPython.

Target: ~1000 Q&A pairs.
Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_lua_upy_dataset.py > /tmp/gen-lua-upy.jsonl
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# 1. Lua Basics
# ---------------------------------------------------------------------------

def gen_lua_basics() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are Lua tables and how do they work?",
        "**Tables** are Lua's only data structure — they serve as arrays, dictionaries, objects, modules, and namespaces.\n\n**Array-style (1-indexed!):**\n```lua\nlocal fruits = {\"apple\", \"banana\", \"cherry\"}\nprint(fruits[1])   -- \"apple\" (Lua arrays start at 1)\nprint(#fruits)     -- 3 (length operator)\n\n-- Iterate array\nfor i, fruit in ipairs(fruits) do\n    print(i, fruit)\nend\n\ntable.insert(fruits, \"date\")       -- append\ntable.insert(fruits, 2, \"apricot\") -- insert at position 2\ntable.remove(fruits, 1)             -- remove first element\n```\n\n**Dictionary-style:**\n```lua\nlocal person = {\n    name = \"Alice\",\n    age = 30,\n    [\"full name\"] = \"Alice Smith\",  -- keys with spaces\n    [42] = \"the answer\",            -- numeric keys\n}\n\nprint(person.name)         -- \"Alice\"\nprint(person[\"full name\"]) -- \"Alice Smith\"\n\n-- Iterate all key-value pairs\nfor key, value in pairs(person) do\n    print(key, value)\nend\n```\n\n**Mixed table:**\n```lua\nlocal mixed = {\n    \"first\",            -- [1] = \"first\"\n    \"second\",           -- [2] = \"second\"\n    name = \"mixed\",     -- [\"name\"] = \"mixed\"\n    [10] = \"ten\",       -- [10] = \"ten\"\n}\n-- #mixed is 2 (only contiguous integer keys from 1)\n```\n\n**Tables are references:**\n```lua\nlocal a = {1, 2, 3}\nlocal b = a       -- b points to the SAME table\nb[1] = 99\nprint(a[1])       -- 99 (modified through b)\n```"
    ))

    pairs.append(msg(
        "How do metatables work in Lua?",
        "**Metatables** customize the behavior of tables — enabling OOP, operator overloading, and custom indexing.\n\n**Setting a metatable:**\n```lua\nlocal t = {}\nlocal mt = {}\nsetmetatable(t, mt)  -- mt is now t's metatable\ngetmetatable(t)      -- returns mt\n```\n\n**Key metamethods:**\n\n**`__index`** — Called when accessing a missing key:\n```lua\nlocal defaults = { color = \"red\", size = 10 }\nlocal obj = setmetatable({}, { __index = defaults })\n\nprint(obj.color)  -- \"red\" (found in defaults)\nobj.color = \"blue\"\nprint(obj.color)  -- \"blue\" (now in obj itself)\n```\n\n**`__newindex`** — Called when setting a new key:\n```lua\nlocal readonly = setmetatable({}, {\n    __newindex = function(t, k, v)\n        error(\"Attempt to modify read-only table\")\n    end\n})\n```\n\n**`__call`** — Make a table callable:\n```lua\nlocal counter = setmetatable({ count = 0 }, {\n    __call = function(self)\n        self.count = self.count + 1\n        return self.count\n    end\n})\nprint(counter())  -- 1\nprint(counter())  -- 2\n```\n\n**OOP with metatables:**\n```lua\nlocal Animal = {}\nAnimal.__index = Animal\n\nfunction Animal.new(name, sound)\n    local self = setmetatable({}, Animal)\n    self.name = name\n    self.sound = sound\n    return self\nend\n\nfunction Animal:speak()\n    print(self.name .. \" says \" .. self.sound)\nend\n\nlocal cat = Animal.new(\"Cat\", \"meow\")\ncat:speak()  -- \"Cat says meow\"\n\n-- Inheritance\nlocal Dog = setmetatable({}, { __index = Animal })\nDog.__index = Dog\n\nfunction Dog.new(name)\n    local self = Animal.new(name, \"woof\")\n    return setmetatable(self, Dog)\nend\n\nfunction Dog:fetch(item)\n    print(self.name .. \" fetches \" .. item)\nend\n\nlocal rex = Dog.new(\"Rex\")\nrex:speak()         -- \"Rex says woof\" (inherited)\nrex:fetch(\"ball\")   -- \"Rex fetches ball\"\n```\n\n**Operator overloading:**\n```lua\nlocal Vec2 = {}\nVec2.__index = Vec2\n\nfunction Vec2.new(x, y)\n    return setmetatable({ x = x, y = y }, Vec2)\nend\n\nVec2.__add = function(a, b)\n    return Vec2.new(a.x + b.x, a.y + b.y)\nend\n\nVec2.__tostring = function(v)\n    return string.format(\"(%g, %g)\", v.x, v.y)\nend\n\nlocal v = Vec2.new(1, 2) + Vec2.new(3, 4)\nprint(v)  -- (4, 6)\n```"
    ))

    pairs.append(msg(
        "How do closures and coroutines work in Lua?",
        "**Closures** capture variables from their enclosing scope:\n```lua\nfunction make_counter(start)\n    local count = start or 0\n    return {\n        increment = function()\n            count = count + 1\n            return count\n        end,\n        get = function()\n            return count\n        end,\n        reset = function()\n            count = 0\n        end,\n    }\nend\n\nlocal c = make_counter(10)\nprint(c.increment())  -- 11\nprint(c.increment())  -- 12\nprint(c.get())        -- 12\n```\n\n**Coroutines** are cooperative multitasking — they yield and resume:\n```lua\nfunction fibonacci()\n    local a, b = 0, 1\n    while true do\n        coroutine.yield(a)\n        a, b = b, a + b\n    end\nend\n\nlocal co = coroutine.create(fibonacci)\nfor i = 1, 10 do\n    local ok, value = coroutine.resume(co)\n    print(value)  -- 0, 1, 1, 2, 3, 5, 8, 13, 21, 34\nend\n\n-- Using coroutine.wrap (simpler API)\nlocal fib = coroutine.wrap(fibonacci)\nfor i = 1, 5 do\n    print(fib())  -- 0, 1, 1, 2, 3\nend\n```\n\n**Producer-consumer with coroutines:**\n```lua\nfunction producer()\n    return coroutine.wrap(function()\n        local items = {\"task1\", \"task2\", \"task3\"}\n        for _, item in ipairs(items) do\n            print(\"Producing: \" .. item)\n            coroutine.yield(item)\n        end\n    end)\nend\n\nfunction consumer(gen)\n    for item in gen do\n        print(\"Consuming: \" .. item)\n    end\nend\n\nconsumer(producer())\n```\n\n**Coroutine states:**\n- `suspended`: Created or after yield\n- `running`: Currently executing\n- `dead`: Finished (returned from function)\n- `normal`: Resumed another coroutine\n\n```lua\nprint(coroutine.status(co))  -- \"suspended\", \"running\", \"dead\"\n```"
    ))

    pairs.append(msg(
        "How do Lua string patterns work?",
        "Lua uses its own pattern language (not regex, but similar and simpler).\n\n**Character classes:**\n| Pattern | Matches |\n|---------|--------|\n| `.` | Any character |\n| `%a` | Letters |\n| `%d` | Digits |\n| `%l` | Lowercase |\n| `%u` | Uppercase |\n| `%w` | Alphanumeric |\n| `%s` | Whitespace |\n| `%p` | Punctuation |\n| `%c` | Control characters |\n| `%A` | NOT letters (uppercase = complement) |\n\n**Quantifiers:**\n| Pattern | Meaning |\n|---------|--------|\n| `*` | 0 or more (greedy) |\n| `+` | 1 or more (greedy) |\n| `-` | 0 or more (lazy) |\n| `?` | 0 or 1 |\n\n**Examples:**\n```lua\nlocal s = \"Hello World 123\"\n\n-- Find\nprint(string.find(s, \"%d+\"))  -- 13, 15 (start, end)\n\n-- Match (returns captures)\nprint(string.match(s, \"(%a+) (%a+)\"))  -- \"Hello\", \"World\"\n\n-- Gmatch (iterator)\nfor word in string.gmatch(s, \"%a+\") do\n    print(word)  -- Hello, World\nend\n\n-- Gsub (replace)\nprint(string.gsub(s, \"%d+\", \"###\"))  -- \"Hello World ###\"\n\n-- Captures\nlocal date = \"2024-03-15\"\nlocal year, month, day = date:match(\"(%d+)-(%d+)-(%d+)\")\nprint(year, month, day)  -- 2024, 03, 15\n\n-- Extract key=value pairs\nlocal config = \"host=localhost port=8080 debug=true\"\nfor key, value in config:gmatch(\"(%w+)=(%w+)\") do\n    print(key, value)\nend\n```\n\n**Differences from regex:**\n- No `|` alternation\n- No `{n,m}` repetition counts\n- Use `%` to escape magic characters: `%.` matches literal dot\n- `[^%d]` = character class negation\n- `-` is lazy (not `*?`)\n- For full regex, use `lrexlib` or LPeg"
    ))

    pairs.append(msg(
        "How do you implement modules in Lua?",
        "**Module pattern (standard):**\n```lua\n-- mymodule.lua\nlocal M = {}  -- module table\n\n-- Private (local to file)\nlocal function helper(x)\n    return x * 2\nend\n\n-- Public (added to module table)\nfunction M.greet(name)\n    print(\"Hello, \" .. name .. \"!\")\nend\n\nfunction M.double(x)\n    return helper(x)\nend\n\n-- Constants\nM.VERSION = \"1.0.0\"\n\nreturn M\n```\n\n**Using the module:**\n```lua\nlocal mymodule = require(\"mymodule\")\nmymodule.greet(\"World\")\nprint(mymodule.double(21))  -- 42\nprint(mymodule.VERSION)     -- \"1.0.0\"\n```\n\n**Module with OOP:**\n```lua\n-- class.lua\nlocal Class = {}\nClass.__index = Class\n\nfunction Class.new(name)\n    return setmetatable({ name = name }, Class)\nend\n\nfunction Class:get_name()\n    return self.name\nend\n\nreturn Class\n```\n\n**`require` behavior:**\n- Searches `package.path` (for `.lua` files) and `package.cpath` (for C modules)\n- **Caches** the result — second `require` returns the same table (no re-execution)\n- Default path: `./?.lua;./?/init.lua;/usr/local/share/lua/5.4/?.lua`\n\n```lua\n-- Check/modify search path\nprint(package.path)\npackage.path = package.path .. \";/my/custom/path/?.lua\"\n\n-- Force reload\npackage.loaded[\"mymodule\"] = nil\nlocal fresh = require(\"mymodule\")\n```\n\n**Init module (directory as module):**\n```\nmylib/\n├── init.lua     -- require(\"mylib\") loads this\n├── utils.lua\n└── config.lua\n```\n```lua\n-- mylib/init.lua\nreturn {\n    utils = require(\"mylib.utils\"),\n    config = require(\"mylib.config\"),\n}\n```"
    ))

    pairs.append(msg(
        "What are the differences between Lua 5.1, 5.3, 5.4, and LuaJIT?",
        "| Feature | Lua 5.1 | Lua 5.3 | Lua 5.4 | LuaJIT 2.1 |\n|---------|---------|---------|---------|------------|\n| **Numbers** | Double only | Integer + float | Integer + float | Double only |\n| **Integers** | N/A | 64-bit | 64-bit | N/A |\n| **Bitwise ops** | Library | Built-in (`&`,`|`,`~`,`<<`,`>>`) | Built-in | `bit` library |\n| **Goto** | No | Yes | Yes | Yes |\n| **Generalized for** | `ipairs`/`pairs` | Same | `<close>` variables | Same as 5.1 |\n| **String length** | 2GB | 2GB | 2GB | 2GB |\n| **GC** | Mark-sweep | Incremental | **Generational** | Incremental |\n| **Speed** | Moderate | Moderate | Moderate | **Very fast** (JIT) |\n| **FFI** | No | No | No | **Yes** (C FFI) |\n| **Environments** | `setfenv` | `_ENV` | `_ENV` | `setfenv` |\n\n**Key version differences:**\n\n**Lua 5.1** (2006): Most embedded, LuaJIT-compatible\n```lua\n-- setfenv for module system (removed in 5.2+)\nmodule(\"mymod\", package.seeall)\n```\n\n**Lua 5.3** (2015): Integer subtype, bitwise operators\n```lua\nprint(type(42))     -- \"number\" but internally integer\nprint(3 // 2)       -- 1 (integer division)\nprint(0xFF & 0x0F)  -- 15 (bitwise AND)\n```\n\n**Lua 5.4** (2020): Generational GC, `<close>` variables\n```lua\nlocal f <close> = io.open(\"file.txt\")\n-- f is automatically closed when it goes out of scope\n```\n\n**LuaJIT** (Lua 5.1 compatible):\n- 10-100x faster than standard Lua for numeric code\n- FFI for calling C functions without bindings:\n```lua\nlocal ffi = require(\"ffi\")\nffi.cdef[[ int printf(const char *fmt, ...); ]]\nffi.C.printf(\"Hello from C!\\n\")\n```\n\n**Use LuaJIT** for: game engines (Defold, LOVE2D), high-performance embedded, FFI needs.\n**Use Lua 5.4** for: new projects without LuaJIT dependency, integer math, generational GC."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. MicroPython
# ---------------------------------------------------------------------------

def gen_micropython() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you use GPIO with MicroPython?",
        """```python
from machine import Pin
import time

# --- Output ---
led = Pin(2, Pin.OUT)  # GPIO 2 as output

led.on()    # or led.value(1)
led.off()   # or led.value(0)
led.toggle()

# Blink LED
while True:
    led.toggle()
    time.sleep_ms(500)

# --- Input ---
button = Pin(0, Pin.IN, Pin.PULL_UP)  # GPIO 0 with internal pull-up

# Poll
if button.value() == 0:  # Pressed (active low with pull-up)
    print("Button pressed!")

# --- Interrupt (IRQ) ---
def button_handler(pin):
    print(f"Button on {pin} pressed!")

button.irq(trigger=Pin.IRQ_FALLING, handler=button_handler)
# IRQ_FALLING = high→low (button press with pull-up)
# IRQ_RISING  = low→high (button release)
# IRQ_FALLING | IRQ_RISING = both edges

# --- Debouncing ---
import time

last_press = 0

def debounced_handler(pin):
    global last_press
    now = time.ticks_ms()
    if time.ticks_diff(now, last_press) > 200:  # 200ms debounce
        last_press = now
        print("Debounced press!")

button.irq(trigger=Pin.IRQ_FALLING, handler=debounced_handler)
```

**Pin modes:**
- `Pin.OUT`: Output (push-pull)
- `Pin.IN`: Input
- `Pin.OPEN_DRAIN`: Open drain output
- `Pin.PULL_UP`: Internal pull-up resistor
- `Pin.PULL_DOWN`: Internal pull-down resistor

**Note:** Available GPIOs vary by board. On ESP32: GPIO 0, 2, 4, 5, 12-19, 21-23, 25-27, 32-39 (34-39 input only)."""
    ))

    pairs.append(msg(
        "How do you use I2C with MicroPython?",
        """```python
from machine import I2C, Pin

# --- Hardware I2C ---
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)  # 400 kHz

# --- Software I2C (any pins) ---
from machine import SoftI2C
i2c = SoftI2C(scl=Pin(22), sda=Pin(21), freq=100000)

# --- Scan for devices ---
devices = i2c.scan()
print(f"Found {len(devices)} devices: {[hex(d) for d in devices]}")
# Example output: Found 2 devices: ['0x3c', '0x68']

# --- Read/Write ---
# Write bytes to device
i2c.writeto(0x3C, b'\\x00\\x01')  # address, data

# Read bytes from device
data = i2c.readfrom(0x3C, 4)  # address, num_bytes

# Write to register, then read
i2c.writeto(0x68, b'\\x75')           # Select register 0x75
who_am_i = i2c.readfrom(0x68, 1)     # Read 1 byte

# Combined write-then-read (more efficient)
i2c.readfrom_mem(0x68, 0x75, 1)      # Read 1 byte from register 0x75
i2c.writeto_mem(0x68, 0x6B, b'\\x00') # Write to register 0x6B

# --- Example: BME280 temperature sensor ---
BME280_ADDR = 0x76

def read_bme280_id(i2c):
    chip_id = i2c.readfrom_mem(BME280_ADDR, 0xD0, 1)
    return chip_id[0]  # Should be 0x60 for BME280

def read_temperature(i2c):
    # Read raw temperature (3 bytes from register 0xFA)
    raw = i2c.readfrom_mem(BME280_ADDR, 0xFA, 3)
    raw_temp = (raw[0] << 12) | (raw[1] << 4) | (raw[2] >> 4)
    # Apply compensation (simplified)
    return raw_temp / 16384.0 * 5  # Placeholder calculation

# --- Example: SSD1306 OLED display ---
# pip install micropython-ssd1306
from ssd1306 import SSD1306_I2C

oled = SSD1306_I2C(128, 64, i2c)
oled.fill(0)           # Clear
oled.text("Hello!", 0, 0)
oled.text("MicroPython", 0, 16)
oled.show()            # Update display
```"""
    ))

    pairs.append(msg(
        "How do you use SPI with MicroPython?",
        """```python
from machine import SPI, Pin

# --- Hardware SPI ---
spi = SPI(1,
    baudrate=10000000,   # 10 MHz
    polarity=0,          # Clock idle low
    phase=0,             # Sample on rising edge
    sck=Pin(18),
    mosi=Pin(23),
    miso=Pin(19),
)

# Chip select (active low, managed manually)
cs = Pin(5, Pin.OUT, value=1)  # Start high (deselected)

# --- Basic read/write ---
def spi_transfer(data):
    cs.value(0)          # Select device
    result = spi.read(len(data), data[0])  # Read while writing
    cs.value(1)          # Deselect
    return result

# Write bytes
cs.value(0)
spi.write(b'\\x01\\x02\\x03')
cs.value(1)

# Read bytes
cs.value(0)
buf = spi.read(4)  # Read 4 bytes
cs.value(1)

# Simultaneous read/write
cs.value(0)
tx = bytearray(b'\\x9F\\x00\\x00\\x00')  # JEDEC ID command
rx = bytearray(4)
spi.write_readinto(tx, rx)
cs.value(1)
print(f"JEDEC ID: {rx[1]:02x} {rx[2]:02x} {rx[3]:02x}")

# --- Example: SPI Flash (W25Q) ---
class SPIFlash:
    def __init__(self, spi, cs_pin):
        self.spi = spi
        self.cs = cs_pin

    def _cmd(self, cmd, read_len=0):
        self.cs.value(0)
        self.spi.write(bytes([cmd]))
        result = self.spi.read(read_len) if read_len else None
        self.cs.value(1)
        return result

    def read_id(self):
        self.cs.value(0)
        self.spi.write(b'\\x9F')
        mfr = self.spi.read(1)[0]
        dev = self.spi.read(2)
        self.cs.value(1)
        return mfr, dev

    def read_status(self):
        return self._cmd(0x05, 1)[0]

flash = SPIFlash(spi, cs)
mfr, dev = flash.read_id()
print(f"Flash: manufacturer={mfr:02x}")
```"""
    ))

    pairs.append(msg(
        "How do you use UART with MicroPython?",
        """```python
from machine import UART, Pin

# --- Initialize UART ---
uart = UART(2,                   # UART number (ESP32: 0, 1, 2)
    baudrate=115200,
    bits=8,
    parity=None,
    stop=1,
    tx=Pin(17),
    rx=Pin(16),
    timeout=1000,                # Read timeout in ms
)

# --- Write ---
uart.write(b"Hello UART!\\n")
uart.write("Temperature: {:.1f}C\\n".format(25.3))

# --- Read ---
# Read available data
data = uart.read()  # Read all available bytes
if data:
    print(f"Received: {data}")

# Read exact number of bytes
data = uart.read(10)  # Read up to 10 bytes (or timeout)

# Read line (until \\n)
line = uart.readline()

# Check if data available
if uart.any():
    data = uart.read(uart.any())

# --- Non-blocking read loop ---
import time

while True:
    if uart.any():
        data = uart.readline()
        if data:
            line = data.decode().strip()
            print(f"Got: {line}")
            # Echo back
            uart.write(f"Echo: {line}\\n")
    time.sleep_ms(10)

# --- GPS module example (NMEA) ---
def parse_gga(sentence):
    \"\"\"Parse $GPGGA NMEA sentence.\"\"\"
    parts = sentence.split(',')
    if parts[0] != '$GPGGA':
        return None
    return {
        'time': parts[1],
        'lat': parts[2],
        'lat_dir': parts[3],
        'lon': parts[4],
        'lon_dir': parts[5],
        'fix': int(parts[6]) if parts[6] else 0,
        'satellites': int(parts[7]) if parts[7] else 0,
    }

gps_uart = UART(2, baudrate=9600, tx=Pin(17), rx=Pin(16))

while True:
    line = gps_uart.readline()
    if line:
        try:
            sentence = line.decode().strip()
            if sentence.startswith('$GPGGA'):
                pos = parse_gga(sentence)
                if pos and pos['fix'] > 0:
                    print(f"Lat: {pos['lat']}{pos['lat_dir']} Lon: {pos['lon']}{pos['lon_dir']}")
        except Exception:
            pass
    time.sleep_ms(100)
```"""
    ))

    pairs.append(msg(
        "How do you use PWM and ADC with MicroPython?",
        """```python
from machine import Pin, PWM, ADC

# ===== PWM (Pulse Width Modulation) =====

# Create PWM on a pin
pwm = PWM(Pin(2))
pwm.freq(1000)       # Set frequency to 1 kHz
pwm.duty(512)        # Set duty cycle (0-1023 on ESP32, 10-bit)
pwm.duty_u16(32768)  # 16-bit duty (0-65535), more portable

# LED brightness control
led_pwm = PWM(Pin(2), freq=5000)

# Fade LED
import time
for duty in range(0, 1024, 10):
    led_pwm.duty(duty)
    time.sleep_ms(20)
for duty in range(1023, -1, -10):
    led_pwm.duty(duty)
    time.sleep_ms(20)

# Servo motor control (50 Hz, 1-2ms pulse)
servo = PWM(Pin(18), freq=50)

def set_servo_angle(angle):
    \"\"\"Set servo angle (0-180 degrees).\"\"\"
    # 1ms pulse = 0 degrees, 2ms pulse = 180 degrees
    # At 50 Hz, period = 20ms
    # duty = pulse_ms / period_ms * max_duty
    min_duty = int(1.0 / 20.0 * 1023)   # ~51\n    max_duty = int(2.0 / 20.0 * 1023)   # ~102\n    duty = min_duty + int((max_duty - min_duty) * angle / 180)\n    servo.duty(duty)\n\nset_servo_angle(90)  # Center\nset_servo_angle(0)   # Full left\nset_servo_angle(180) # Full right\n\n# Buzzer / tone\nbuzzer = PWM(Pin(25), freq=440, duty=512)  # A4 note\ntime.sleep(0.5)\nbuzzer.deinit()  # Stop\n\n# ===== ADC (Analog to Digital Converter) =====\n\nadc = ADC(Pin(34))  # GPIO 34 (ESP32: 32-39 are ADC-capable)\nadc.atten(ADC.ATTN_11DB)   # Full range: 0-3.3V\nadc.width(ADC.WIDTH_12BIT) # 12-bit resolution (0-4095)\n\n# Read raw value\nraw = adc.read()           # 0-4095\nraw_u16 = adc.read_u16()   # 0-65535 (portable)\n\n# Convert to voltage\nvoltage = raw / 4095.0 * 3.3\nprint(f\"Voltage: {voltage:.2f}V\")\n\n# Read temperature sensor (NTC thermistor example)\nimport math\n\ndef read_temperature(adc_pin, r_fixed=10000, beta=3950, t0=298.15, r0=10000):\n    raw = adc_pin.read()\n    if raw == 0:\n        return None\n    voltage = raw / 4095.0 * 3.3\n    r_ntc = r_fixed * voltage / (3.3 - voltage)\n    temp_k = 1.0 / (1.0/t0 + (1.0/beta) * math.log(r_ntc/r0))\n    return temp_k - 273.15  # Celsius\n\ntemp = read_temperature(adc)\nprint(f\"Temperature: {temp:.1f}C\")\n```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. ESP32 MicroPython
# ---------------------------------------------------------------------------

def gen_esp32_micropython() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you connect to WiFi with MicroPython on ESP32?",
        """```python
import network
import time

def connect_wifi(ssid, password, timeout=10):
    \"\"\"Connect to WiFi and return the interface.\"\"\"
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        print(f"Already connected: {wlan.ifconfig()}")
        return wlan

    print(f"Connecting to {ssid}...")
    wlan.connect(ssid, password)

    start = time.time()
    while not wlan.isconnected():
        if time.time() - start > timeout:
            raise OSError("WiFi connection timeout")
        time.sleep(0.5)

    ip, subnet, gateway, dns = wlan.ifconfig()
    print(f"Connected! IP: {ip}")
    return wlan

# --- Access Point mode ---
def start_ap(ssid="ESP32-AP", password="12345678"):
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=ssid, password=password, authmode=network.AUTH_WPA2_PSK)
    print(f"AP started: {ap.ifconfig()}")
    return ap

# --- Scan networks ---
def scan_networks():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    networks = wlan.scan()
    for ssid, bssid, channel, rssi, authmode, hidden in networks:
        print(f"  {ssid.decode():30s} CH:{channel:2d} RSSI:{rssi}dBm")

# --- Auto-reconnect ---
def ensure_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    if not wlan.isconnected():
        print("WiFi disconnected, reconnecting...")
        connect_wifi(ssid, password)
    return wlan

# --- HTTP request (urequests) ---
import urequests

wlan = connect_wifi("MySSID", "MyPassword")
response = urequests.get("http://api.example.com/data")
print(response.json())
response.close()

# POST request
response = urequests.post(
    "http://api.example.com/sensor",
    json={"temperature": 25.3, "humidity": 60},
    headers={"Content-Type": "application/json"},
)
print(response.status_code)
response.close()
```"""
    ))

    pairs.append(msg(
        "How do you set up a web server on ESP32 with MicroPython?",
        """```python
import network
import socket
import json

def start_webserver(port=80):
    \"\"\"Simple HTTP server on ESP32.\"\"\"
    addr = socket.getaddrinfo('0.0.0.0', port)[0][-1]
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(addr)
    s.listen(5)
    print(f"Server listening on port {port}")

    while True:
        cl, addr = s.accept()
        print(f"Client connected from {addr}")
        try:
            request = cl.recv(1024).decode()
            method, path, _ = request.split(' ', 2)
            print(f"{method} {path}")

            if path == '/':
                response = html_page()
                cl.send(http_response(response, 'text/html'))

            elif path == '/api/sensor':
                data = read_sensors()
                cl.send(http_response(
                    json.dumps(data), 'application/json'
                ))

            elif path == '/api/led/on':
                led.on()
                cl.send(http_response('{\"led\": \"on\"}', 'application/json'))

            elif path == '/api/led/off':
                led.off()
                cl.send(http_response('{\"led\": \"off\"}', 'application/json'))

            else:
                cl.send(http_response('Not Found', 'text/plain', 404))

        except Exception as e:
            print(f"Error: {e}")
        finally:
            cl.close()

def http_response(body, content_type, status=200):
    status_text = {200: 'OK', 404: 'Not Found', 500: 'Error'}
    return (
        f\"HTTP/1.1 {status} {status_text.get(status, 'OK')}\\r\\n\"
        f\"Content-Type: {content_type}\\r\\n\"
        f\"Content-Length: {len(body)}\\r\\n\"
        f\"Connection: close\\r\\n\"
        f\"Access-Control-Allow-Origin: *\\r\\n\"
        f\"\\r\\n{body}\"
    ).encode()

def html_page():
    return \"\"\"<!DOCTYPE html>
<html>
<head><title>ESP32</title></head>
<body>
  <h1>ESP32 Control</h1>
  <button onclick=\"fetch('/api/led/on')\">LED ON</button>
  <button onclick=\"fetch('/api/led/off')\">LED OFF</button>
  <div id=\"sensor\"></div>
  <script>
    setInterval(async () => {
      const r = await fetch('/api/sensor');
      document.getElementById('sensor').textContent = JSON.stringify(await r.json());
    }, 2000);
  </script>
</body>
</html>\"\"\"

from machine import Pin
led = Pin(2, Pin.OUT)

def read_sensors():
    return {\"temperature\": 25.3, \"humidity\": 60, \"led\": led.value()}

# Connect to WiFi first, then start server
connect_wifi(\"MySSID\", \"MyPassword\")
start_webserver()
```"""
    ))

    pairs.append(msg(
        "How do you handle the filesystem and OTA updates in MicroPython?",
        """**Filesystem operations:**
```python
import os

# List files
print(os.listdir('/'))        # Root directory
print(os.listdir('/lib'))     # Lib directory

# File I/O
with open('/data.txt', 'w') as f:
    f.write('Hello from MicroPython\\n')
    f.write('Line 2\\n')

with open('/data.txt', 'r') as f:
    for line in f:
        print(line.strip())

# JSON config file
import json

def save_config(config):
    with open('/config.json', 'w') as f:
        json.dump(config, f)

def load_config():
    try:
        with open('/config.json', 'r') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {'ssid': '', 'password': '', 'interval': 60}

# Directory operations
os.mkdir('/data')
os.rename('/data.txt', '/data/log.txt')
os.remove('/data/log.txt')
os.rmdir('/data')

# Filesystem info
stat = os.statvfs('/')
block_size = stat[0]
total_blocks = stat[2]
free_blocks = stat[3]
print(f\"Total: {total_blocks * block_size // 1024}KB\")
print(f\"Free: {free_blocks * block_size // 1024}KB\")
```

**OTA (Over-The-Air) updates:**
```python
import urequests
import os
import machine

OTA_URL = \"http://myserver.com/firmware\"

def ota_update():
    \"\"\"Download and apply OTA update.\"\"\"
    # Check for update
    response = urequests.get(f\"{OTA_URL}/version.txt\")
    remote_version = response.text.strip()
    response.close()

    try:
        with open('/version.txt', 'r') as f:
            local_version = f.read().strip()
    except OSError:
        local_version = \"0.0.0\"

    if remote_version == local_version:
        print(\"Already up to date\")
        return False

    print(f\"Updating {local_version} -> {remote_version}\")

    # Download files
    files = ['main.py', 'config.py', 'lib/sensor.py']
    for filename in files:
        print(f\"Downloading {filename}...\")
        response = urequests.get(f\"{OTA_URL}/{filename}\")
        if response.status_code == 200:
            # Ensure directory exists
            parts = filename.rsplit('/', 1)
            if len(parts) > 1:
                try:
                    os.mkdir(parts[0])
                except OSError:
                    pass
            with open(filename, 'w') as f:
                f.write(response.text)
        response.close()

    # Save new version
    with open('/version.txt', 'w') as f:
        f.write(remote_version)

    print(\"Update complete, rebooting...\")
    machine.reset()

# boot.py — check for updates on startup
# ota_update()
```

**WebREPL** (remote REPL over WebSocket):
```python
import webrepl
webrepl.start(password='mypass')
# Access via http://micropython.org/webrepl/
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. Lua in Embedded
# ---------------------------------------------------------------------------

def gen_lua_embedded() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How is Lua used in embedded systems and IoT?",
        "Lua is popular in embedded systems due to its tiny footprint (~100KB), fast VM, and easy C integration.\n\n**NodeMCU** (ESP8266/ESP32):\n```lua\n-- WiFi connection\nwifi.setmode(wifi.STATION)\nwifi.sta.config({ ssid = \"MyNetwork\", pwd = \"password\" })\nwifi.sta.connect()\n\nwifi.eventmon.register(wifi.eventmon.STA_GOT_IP, function(T)\n    print(\"IP: \" .. T.IP)\nend)\n\n-- GPIO\ngpio.mode(4, gpio.OUTPUT)\ngpio.write(4, gpio.HIGH)\n\n-- Timer (blink LED)\ntmr.create():alarm(1000, tmr.ALARM_AUTO, function()\n    local level = gpio.read(4)\n    gpio.write(4, 1 - level)  -- Toggle\nend)\n\n-- HTTP server\nsrv = net.createServer(net.TCP)\nsrv:listen(80, function(conn)\n    conn:on(\"receive\", function(sck, data)\n        sck:send(\"HTTP/1.1 200 OK\\r\\nContent-Type: text/html\\r\\n\\r\\n\")\n        sck:send(\"<h1>Hello from NodeMCU!</h1>\")\n    end)\nend)\n\n-- MQTT\nm = mqtt.Client(\"esp8266\", 120)\nm:connect(\"broker.hivemq.com\", 1883, false, function(client)\n    print(\"MQTT connected\")\n    client:subscribe(\"home/sensor/#\", 0)\nend)\n\nm:on(\"message\", function(client, topic, data)\n    print(topic .. \": \" .. data)\nend)\n\nm:publish(\"home/sensor/temp\", \"25.3\", 0, 0)\n```\n\n**eLua** (Embedded Lua):\n- Runs on ARM Cortex-M, AVR32, STM32\n- Bare-metal or RTOS (FreeRTOS)\n- Direct hardware register access\n\n**Lua RTOS** (ESP32):\n- Full RTOS integration\n- Hardware abstraction: GPIO, I2C, SPI, UART, ADC, PWM\n- Built-in: WiFi, BLE, LoRa, CAN, MQTT, HTTP\n\n**Why Lua for embedded:**\n- Tiny RAM footprint (~20KB for VM)\n- No OS dependency\n- C API is simple (20 functions)\n- Hot-reload scripts without reflashing\n- Safe sandboxing (restrict API access)"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. LOVE2D
# ---------------------------------------------------------------------------

def gen_love2d() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you make a simple game with LOVE2D?",
        """**LOVE2D** (LOVE) is a Lua game framework — fast prototyping, great for 2D games.

```lua
-- main.lua
-- Simple platformer character

local player = {
    x = 400,
    y = 300,
    width = 32,
    height = 48,
    speed = 200,
    vy = 0,        -- vertical velocity
    gravity = 800,
    jump_force = -400,
    on_ground = false,
}

local platforms = {
    { x = 0, y = 550, w = 800, h = 50 },  -- ground
    { x = 200, y = 400, w = 150, h = 20 },
    { x = 500, y = 300, w = 150, h = 20 },
}

function love.load()
    love.window.setTitle("LOVE2D Platformer")
    love.window.setMode(800, 600)
end

function love.update(dt)
    -- Horizontal movement
    if love.keyboard.isDown("left") or love.keyboard.isDown("a") then
        player.x = player.x - player.speed * dt
    end
    if love.keyboard.isDown("right") or love.keyboard.isDown("d") then
        player.x = player.x + player.speed * dt
    end

    -- Gravity
    player.vy = player.vy + player.gravity * dt
    player.y = player.y + player.vy * dt
    player.on_ground = false

    -- Platform collision
    for _, p in ipairs(platforms) do
        if player.x + player.width > p.x and
           player.x < p.x + p.w and
           player.y + player.height > p.y and
           player.y + player.height < p.y + p.h + player.vy * dt then
            player.y = p.y - player.height
            player.vy = 0
            player.on_ground = true
        end
    end
end

function love.keypressed(key)
    if key == "space" and player.on_ground then
        player.vy = player.jump_force
    end
    if key == "escape" then
        love.event.quit()
    end
end

function love.draw()
    -- Draw platforms
    love.graphics.setColor(0.5, 0.5, 0.5)
    for _, p in ipairs(platforms) do
        love.graphics.rectangle("fill", p.x, p.y, p.w, p.h)
    end

    -- Draw player
    love.graphics.setColor(0.2, 0.7, 1.0)
    love.graphics.rectangle("fill", player.x, player.y,
                            player.width, player.height)

    -- HUD
    love.graphics.setColor(1, 1, 1)
    love.graphics.print("WASD/Arrows to move, Space to jump", 10, 10)
    love.graphics.print(string.format("FPS: %d", love.timer.getFPS()), 10, 30)
end
```

**Run:** `love .` (from the directory containing main.lua)

**LOVE2D callbacks:**
- `love.load()`: Initialize (once)
- `love.update(dt)`: Game logic (every frame, dt = delta time)
- `love.draw()`: Rendering (every frame)
- `love.keypressed(key)`: Key press event
- `love.mousepressed(x, y, button)`: Mouse click
- `love.conf(t)`: Configuration (window size, modules)"""
    ))

    pairs.append(msg(
        "How do you handle sprites, animation, and audio in LOVE2D?",
        """```lua
-- Sprite loading and animation
local spritesheet = love.graphics.newImage("player_sheet.png")
local frame_width, frame_height = 32, 48
local frames = {}
local current_frame = 1
local anim_timer = 0
local anim_speed = 0.1  -- seconds per frame

function love.load()
    -- Cut spritesheet into quads
    local sheet_w = spritesheet:getWidth()
    local sheet_h = spritesheet:getHeight()
    local cols = sheet_w / frame_width

    for i = 0, cols - 1 do
        frames[#frames + 1] = love.graphics.newQuad(
            i * frame_width, 0,
            frame_width, frame_height,
            sheet_w, sheet_h
        )
    end

    -- Audio
    music = love.audio.newSource("music.ogg", "stream")  -- stream for long audio
    music:setLooping(true)
    music:setVolume(0.5)
    music:play()

    jump_sfx = love.audio.newSource("jump.wav", "static")  -- static for short SFX
    hit_sfx = love.audio.newSource("hit.wav", "static")
end

function love.update(dt)
    -- Animate sprite
    anim_timer = anim_timer + dt
    if anim_timer >= anim_speed then
        anim_timer = anim_timer - anim_speed
        current_frame = current_frame % #frames + 1
    end
end

function love.draw()
    -- Draw animated sprite (with scaling and rotation)
    love.graphics.draw(
        spritesheet,
        frames[current_frame],
        player.x, player.y,  -- position
        0,                    -- rotation
        2, 2,                 -- scale x, y
        frame_width/2,        -- origin x (center)
        frame_height/2        -- origin y (center)
    )
end

function love.keypressed(key)
    if key == "space" then
        jump_sfx:stop()   -- Reset if already playing
        jump_sfx:play()
    end
end

-- Particle system
local particles

function love.load()
    local img = love.graphics.newImage("particle.png")
    particles = love.graphics.newParticleSystem(img, 100)
    particles:setEmissionRate(20)
    particles:setParticleLifetime(0.5, 1.5)
    particles:setSpeed(50, 150)
    particles:setSpread(math.pi * 2)  -- 360 degrees
    particles:setColors(1, 0.5, 0, 1,  1, 0, 0, 0)  -- orange to transparent red
    particles:setSizes(1, 0.5, 0)  -- shrink over lifetime
end

function love.update(dt)
    particles:setPosition(player.x, player.y)
    particles:update(dt)
end

function love.draw()
    love.graphics.draw(particles)
end
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. Neovim Lua
# ---------------------------------------------------------------------------

def gen_neovim_lua() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you configure Neovim with Lua?",
        """Neovim supports Lua configuration via `init.lua` (replaces `init.vim`).

**~/.config/nvim/init.lua:**
```lua
-- Basic settings
vim.g.mapleader = ' '  -- Leader key = space
vim.g.maplocalleader = ' '

vim.opt.number = true           -- Line numbers
vim.opt.relativenumber = true   -- Relative line numbers
vim.opt.mouse = 'a'             -- Mouse support
vim.opt.tabstop = 4             -- Tab width
vim.opt.shiftwidth = 4          -- Indent width
vim.opt.expandtab = true        -- Spaces instead of tabs
vim.opt.smartindent = true
vim.opt.wrap = false
vim.opt.swapfile = false
vim.opt.backup = false
vim.opt.undofile = true
vim.opt.hlsearch = false        -- Don't highlight search
vim.opt.incsearch = true        -- Incremental search
vim.opt.termguicolors = true
vim.opt.scrolloff = 8           -- Keep 8 lines visible
vim.opt.signcolumn = 'yes'
vim.opt.clipboard = 'unnamedplus'  -- System clipboard

-- Keymaps
local keymap = vim.keymap.set

-- Normal mode
keymap('n', '<leader>w', ':w<CR>', { desc = 'Save' })
keymap('n', '<leader>q', ':q<CR>', { desc = 'Quit' })
keymap('n', '<C-d>', '<C-d>zz', { desc = 'Scroll down centered' })
keymap('n', '<C-u>', '<C-u>zz', { desc = 'Scroll up centered' })

-- Window navigation
keymap('n', '<C-h>', '<C-w>h')
keymap('n', '<C-j>', '<C-w>j')
keymap('n', '<C-k>', '<C-w>k')
keymap('n', '<C-l>', '<C-w>l')

-- Visual mode: move lines
keymap('v', 'J', \":m '>+1<CR>gv=gv\")
keymap('v', 'K', \":m '<-2<CR>gv=gv\")

-- Autocommands
vim.api.nvim_create_autocmd('TextYankPost', {
    callback = function()
        vim.highlight.on_yank({ timeout = 200 })
    end,
})

-- File type specific
vim.api.nvim_create_autocmd('FileType', {
    pattern = { 'lua', 'yaml', 'json' },
    callback = function()
        vim.opt_local.tabstop = 2
        vim.opt_local.shiftwidth = 2
    end,
})

-- Plugin manager (lazy.nvim)
local lazypath = vim.fn.stdpath('data') .. '/lazy/lazy.nvim'
if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({ 'git', 'clone', '--branch=stable',
        'https://github.com/folke/lazy.nvim.git', lazypath })
end
vim.opt.rtp:prepend(lazypath)

require('lazy').setup({
    'nvim-treesitter/nvim-treesitter',
    'neovim/nvim-lspconfig',
    'hrsh7th/nvim-cmp',
    { 'catppuccin/nvim', name = 'catppuccin' },
    'nvim-telescope/telescope.nvim',
})
```

**Key Neovim Lua APIs:**
- `vim.opt`: Set options (`vim.opt.number = true`)
- `vim.g`: Global variables (`vim.g.mapleader`)
- `vim.keymap.set`: Key mappings
- `vim.api`: Raw API access
- `vim.fn`: Call Vimscript functions
- `vim.cmd`: Execute Vim commands"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. MicroPython vs CircuitPython
# ---------------------------------------------------------------------------

def gen_upy_vs_cpy() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the differences between MicroPython and CircuitPython?",
        "| Feature | MicroPython | CircuitPython |\n|---------|-------------|---------------|\n| **Maintainer** | Damien George | Adafruit |\n| **Origin** | Original project (2013) | Fork of MicroPython (2017) |\n| **Focus** | General embedded Python | Education, ease of use |\n| **Boards** | ESP32, STM32, RP2040, many | Adafruit boards, RP2040, ESP32-S3 |\n| **USB** | Basic | **USB mass storage** (shows as drive) |\n| **Libraries** | Community packages | **200+ Adafruit libraries** (unified) |\n| **REPL** | Serial | Serial + **USB serial** (auto-detected) |\n| **Workflow** | Upload via ampy/mpremote/WebREPL | **Drag and drop** files to CIRCUITPY drive |\n| **WiFi** | `network` module | `wifi` + `socketpool` modules |\n| **BLE** | `bluetooth` module | `_bleio` module |\n| **Threading** | `_thread` module (limited) | **No threading** (single-threaded) |\n| **Interrupts** | `Pin.irq()` | Limited (asyncio preferred) |\n| **Performance** | Generally faster | Slightly slower (safety checks) |\n| **code.py** | `main.py` on boot | `code.py` on boot (auto-reload on save) |\n\n**CircuitPython advantages:**\n- Drag-and-drop deployment (appears as USB drive)\n- Unified library bundle (no pip/upip)\n- Better error messages\n- Auto-reload when files change\n- Extensive learning resources from Adafruit\n\n**MicroPython advantages:**\n- More boards supported (STM32, ESP32, WiPy, etc.)\n- Threading support\n- Lower-level hardware access\n- More mature networking stack\n- Closer to CPython stdlib\n- Better for production embedded\n\n**CircuitPython GPIO example:**\n```python\nimport board\nimport digitalio\nimport time\n\n# board module has pin names (not numbers)\nled = digitalio.DigitalInOut(board.LED)\nled.direction = digitalio.Direction.OUTPUT\n\nbutton = digitalio.DigitalInOut(board.BUTTON)\nbutton.direction = digitalio.Direction.INPUT\nbutton.pull = digitalio.Pull.UP\n\nwhile True:\n    led.value = not button.value  # LED on when button pressed\n    time.sleep(0.01)\n```\n\n**Choose CircuitPython** for: education, prototyping, Adafruit hardware.\n**Choose MicroPython** for: production, ESP32, threading, advanced networking."
    ))

    pairs.append(msg(
        "How do you use asyncio in MicroPython for concurrent tasks?",
        """```python
import asyncio
from machine import Pin, ADC
import time

# --- Basic async task ---
async def blink_led(pin_num, interval_ms):
    led = Pin(pin_num, Pin.OUT)
    while True:
        led.toggle()
        await asyncio.sleep_ms(interval_ms)

# --- Sensor reading task ---
async def read_sensors(interval_s=5):
    adc = ADC(Pin(34))
    adc.atten(ADC.ATTN_11DB)
    while True:
        value = adc.read()
        voltage = value / 4095 * 3.3
        print(f"Sensor: {voltage:.2f}V")
        await asyncio.sleep(interval_s)

# --- WiFi watchdog ---
async def wifi_watchdog(ssid, password, check_interval=30):
    import network
    wlan = network.WLAN(network.STA_IF)
    while True:
        if not wlan.isconnected():
            print("WiFi lost, reconnecting...")
            wlan.active(True)
            wlan.connect(ssid, password)
            for _ in range(20):
                if wlan.isconnected():
                    print(f"Reconnected: {wlan.ifconfig()[0]}")
                    break
                await asyncio.sleep(1)
        await asyncio.sleep(check_interval)

# --- HTTP server with asyncio ---
async def handle_client(reader, writer):
    request = await reader.read(1024)
    request_line = request.decode().split('\\r\\n')[0]
    print(f"Request: {request_line}")

    response = "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\n\\r\\nHello!"
    writer.write(response.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()

async def start_server(port=80):
    server = await asyncio.start_server(handle_client, "0.0.0.0", port)
    print(f"Server on port {port}")
    while True:
        await asyncio.sleep(3600)  # Keep server running

# --- Main: run all tasks concurrently ---
async def main():
    await asyncio.gather(
        blink_led(2, 500),             # Blink LED every 500ms
        read_sensors(5),               # Read sensors every 5s
        wifi_watchdog("SSID", "PASS"), # Check WiFi every 30s
        start_server(80),              # HTTP server
    )

asyncio.run(main())
```

**Key asyncio patterns in MicroPython:**
- `asyncio.sleep_ms()` for millisecond precision
- `asyncio.gather()` to run multiple tasks
- `asyncio.Event()` for task synchronization
- `asyncio.Lock()` for shared resource access
- Stream-based server with `asyncio.start_server()`
- No threading needed — cooperative multitasking via await"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    generators = {
        "Lua Basics": gen_lua_basics,
        "MicroPython": gen_micropython,
        "ESP32 MicroPython": gen_esp32_micropython,
        "Lua Embedded": gen_lua_embedded,
        "LOVE2D": gen_love2d,
        "Neovim Lua": gen_neovim_lua,
        "MicroPython vs CircuitPython": gen_upy_vs_cpy,
    }

    total = 0
    for label, gen_fn in generators.items():
        pairs = gen_fn()
        count = len(pairs)
        total += count
        logger.info("  %-25s %5d pairs", label, count)
        for pair in pairs:
            print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== TOTAL: %d pairs ===", total)


if __name__ == "__main__":
    main()
