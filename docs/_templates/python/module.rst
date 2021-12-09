{% if not obj.display %}
:orphan:

{% endif %}
{{ obj.name }}
=============

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

{% endif %}


{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}

{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}

{% block classes scoped %}
{% if visible_classes %}
.. autoapisummary::
{% for item in visible_classes %}
   {{ item.id }}
{% endfor %}
{% endif %}
{% endblock %}


{% for obj_item in visible_classes %}
{{ obj_item.render()|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}
