import re

from postal.parser import parse_address

from address import Address


class AddressProcessor:
    road_regexp = '([a-zA-ZäöüÄÖÜ]+(( |-|!.)?[a-zA-ZäöüÄÖÜß]+)+)\.?'
    house_number_regexp = '[1-9][0-9]*(( )?[a-zA-Z])?'
    postcode_regexp = '^(?!01000|99999)(0[1-9]\d{3}|[1-9]\d{4})$'
    city_regexp = '([a-zA-ZäöüÄÖÜ]+(( |-)?[a-zA-ZäöüÄÖÜß().]+)+)'

    def __init__(self):
        self.empty_members = []
        self.address = Address('', '', '', '')

    def process_address_input(self, input):
        road, house_number, postcode, city = self.__preprocess_input(input)
        self.address = Address(road, house_number, postcode, city)

        self.empty_members = self.__check_empty_address_members(self.address)
        self.address = self.__match_address_members_from_input(self.address, input)

    # Destructuring of tuples
    def __preprocess_input(self, input):
        tuples = parse_address(input)
        print(tuples)

        road = ''
        house_number = ''
        postcode = ''
        city = ''

        for tuple in tuples:
            key = tuple[1]
            value = tuple[0]

            if key == 'road':
                road = value
            if key == 'house_number':
                house_number = value
            if key == 'postcode':
                postcode = value
            if key == 'city':
                city = value

        return road.strip(), house_number.strip(), postcode.strip(), city.strip()

    def __check_empty_address_members(self, address):
        empty_members = []

        if not address.road:
            empty_members.append('road')
        if not address.house_number:
            empty_members.append('house_number')
        if not address.postcode:
            empty_members.append('postcode')
        if not address.city:
            empty_members.append('city')

        return empty_members

    def __match_address_members_from_input(self, address, input):
        if address.road:
            road_match = self.__match_by_string(address.road, input)
            address.road = road_match.group().strip()

        if address.house_number:
            house_number_match = self.__match_by_string(address.house_number, input)
            address.house_number = house_number_match.group().strip()

        if address.postcode:
            postcode_match = self.__match_by_string(address.postcode, input)
            address.postcode = postcode_match.group().strip()

        if address.city:
            city_match = self.__match_by_string(address.city, input)
            address.city = city_match.group().strip()

        return address

    def __match_by_string(self, string, text):
        string = string.replace(' ', '\\b.?.?\\b')
        regex = fr'\b{string}\b.?'
        print(regex)
        return re.search(regex, text, flags=re.IGNORECASE)
