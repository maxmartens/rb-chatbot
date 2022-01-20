import re

from logger import Logger

from postal.parser import parse_address

from address import Address


class AddressProcessor:
    address_member_labels = {
        'road': 'Road',
        'house_number': 'House Number',
        'postcode': 'Postcode',
        'city': 'City'
    }

    road_key = 'road'
    house_number_key = 'house_number'
    postcode_key = 'postcode'
    city_key = 'city'

    house_number_regexp = r'[1-9][0-9]*(\s?\w(?!\w))?'
    postcode_regexp = r'(?<!\d)\d{5}(?!\d)'

    def __init__(self):
        self.empty_members = []
        self.address = Address('', '', '', '')

    def process_address_input(self, inp):
        Logger.debug(2, 'Process address input:', inp)
        road, house_number, postcode, city = self.__preprocess_input(inp)
        Logger.debug(2, 'Preprocessing result:', road, house_number, postcode, city)
        self.address = Address(road, house_number, postcode, city)

        self.empty_members = self.__get_empty_address_members(self.address)
        Logger.debug(2, 'Empty members:', self.empty_members)
        self.address = self.__match_address_members_from_input(self.address, inp)
        Logger.debug(2, 'Address:', self.address.road, self.address.house_number, self.address.postcode, self.address.city)

    def reprocess_address(self, address):
        address_string = f'{address.road} {address.house_number} {address.postcode} {address.city}'
        Logger.debug(2, 'Reprocess address:', address_string)

        self.process_address_input(address_string)

    # Destructuring of tuples
    def __preprocess_input(self, inp):
        Logger.debug(2, 'Preprocess input:', inp)
        tuples = parse_address(inp)
        Logger.debug(2, 'Tuples:', tuples)

        tuple_count = {
            AddressProcessor.road_key: 0,
            AddressProcessor.house_number_key: 0,
            AddressProcessor.postcode_key: 0,
            AddressProcessor.city_key: 0
        }

        member_values = {
            AddressProcessor.road_key: '',
            AddressProcessor.house_number_key: '',
            AddressProcessor.postcode_key: '',
            AddressProcessor.city_key: ''
        }

        for tuple in tuples:
            key = tuple[1]
            value = tuple[0]

            count = tuple_count.get(key)
            if type(count) is int:
                tuple_count[key] = int(count) + 1

            if type(member_values.get(key)) is str:
                member_values[key] = value

        Logger.debug(2, 'Count of each tuple type:', tuple_count)
        for key in tuple_count:
            count = tuple_count[key]
            if count != 1:
                Logger.debug(2, 'Nulling member:', key)
                member_values[key] = ''

        return member_values[AddressProcessor.road_key].strip(), member_values[AddressProcessor.house_number_key].strip(), \
               member_values[AddressProcessor.postcode_key].strip(), member_values[AddressProcessor.city_key].strip()

    def __get_empty_address_members(self, address):
        empty_members = []

        if not address.road:
            empty_members.append(AddressProcessor.road_key)
        if not address.house_number:
            empty_members.append(AddressProcessor.house_number_key)
        if not address.postcode:
            empty_members.append(AddressProcessor.postcode_key)
        if not address.city:
            empty_members.append(AddressProcessor.city_key)

        return empty_members

    def __match_address_members_from_input(self, address, inp):
        Logger.debug(2, 'Match address members from input:', inp)
        if address.road:
            road_match = self.__match_road_by_string(address.road, inp)
            if road_match:
                Logger.debug(2, 'Matched road:', road_match.group())
                address.road = road_match.group().strip()
            else:
                Logger.debug(2, 'No match for road')
                self.empty_members.append(AddressProcessor.road_key)

        if address.house_number:
            house_number_match = self.__match_house_number(address.house_number)
            if house_number_match:
                Logger.debug(2, 'Matched house number:', house_number_match.group())
                address.house_number = house_number_match.group().strip()
            else:
                Logger.debug(2, 'No match for house number')
                self.empty_members.append(AddressProcessor.house_number_key)

        if address.postcode:
            postcode_match = self.__match_postcode(address.postcode)
            if postcode_match:
                Logger.debug(2, 'Matched postcode:', postcode_match.group())
                address.postcode = postcode_match.group()
            else:
                Logger.debug(2, 'No match for postcode')
                self.empty_members.append(AddressProcessor.postcode_key)

        if address.city:
            city_match = self.__match_city_by_string(address.city, inp)
            if city_match:
                Logger.debug(2, 'Matched city:', city_match.group())
                address.city = city_match.group().strip()
            else:
                Logger.debug(2, 'No match for city')
                self.empty_members.append(AddressProcessor.city_key)

        return address

    def __match_house_number(self, house_number):
        return re.search(AddressProcessor.house_number_regexp, house_number, flags=re.IGNORECASE)

    def __match_postcode(self, postcode):
        return re.search(AddressProcessor.postcode_regexp, postcode)

    def __match_road_by_string(self, string, text):
        string = string.replace('.', '')
        string = string.replace(' ', r'\b.?.?\b')
        regex = fr'\b{string}\b.?'

        Logger.debug(2, 'Match road by string (Regex):', regex)
        return re.search(regex, text, flags=re.IGNORECASE)

    def __match_city_by_string(self, string, text):
        string = string.replace('.', '')
        string = string.replace(' ', r'\b.?.?\b')
        regex = fr'\b{string}\b'

        Logger.debug(2, 'Match city by string (Regex):', regex)
        return re.search(regex, text, flags=re.IGNORECASE)
