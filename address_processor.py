import re

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

    def process_address_input(self, input):
        road, house_number, postcode, city = self.__preprocess_input(input)
        self.address = Address(road, house_number, postcode, city)

        self.empty_members = self.__check_empty_address_members(self.address)
        self.address = self.__match_address_members_from_input(self.address, input)

    def reprocess_address(self, address):
        address_string = f'{address.road} {address.house_number} {address.postcode} {address.city}'
        self.process_address_input(address_string)

    # Destructuring of tuples
    def __preprocess_input(self, input):
        tuples = parse_address(input)
        # print(tuples)

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

        # Member mit mehreren counts nullen
        for key in tuple_count:
            count = tuple_count[key]
            if count != 1:
                member_values[key] = ''

        road = member_values[AddressProcessor.road_key]
        house_number = member_values[AddressProcessor.house_number_key]
        postcode = member_values[AddressProcessor.postcode_key]
        city = member_values[AddressProcessor.city_key]

        return road.strip(), house_number.strip(), postcode.strip(), city.strip()

    def __check_empty_address_members(self, address):
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

    def __match_address_members_from_input(self, address, input):
        if address.road:
            road_match = self.__match_road_by_string(address.road, input)
            if road_match:
                address.road = road_match.group().strip()
            else:
                self.empty_members.append(AddressProcessor.road_key)

        if address.house_number:
            house_number_match = self.__match_house_number(address.house_number)
            if house_number_match:
                address.house_number = house_number_match.group().strip()
            else:
                self.empty_members.append(AddressProcessor.house_number_key)

        if address.postcode:
            postcode_match = self.__match_postcode(address.postcode)
            if postcode_match:
                address.postcode = postcode_match.group()
            else:
                self.empty_members.append(AddressProcessor.postcode_key)

        if address.city:
            city_match = self.__match_city_by_string(address.city, input)
            if city_match:
                address.city = city_match.group().strip()
            else:
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
        # print('[DEBUG] Match String (Regex):', regex)
        return re.search(regex, text, flags=re.IGNORECASE)

    def __match_city_by_string(self, string, text):
        string = string.replace('.', '')
        string = string.replace(' ', r'\b.?.?\b')
        regex = fr'\b{string}\b'
        # print('[DEBUG] Match String (Regex):', regex)
        return re.search(regex, text, flags=re.IGNORECASE)
