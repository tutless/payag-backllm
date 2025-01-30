from marshmallow import Schema, fields


class PlainQuerySchema(Schema):
    query = fields.Str(required=True)


class PlainTestSchema(Schema):
    test = fields.Str(required=True)
