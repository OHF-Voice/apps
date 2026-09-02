# Changelog

## 1.1.0

- Add a "Create calendar event" blueprint
- Restrict entity fields written with the `domain` shorthand
  (`entity: {domain: calendar}`) to that domain, instead of offering every
  exposed entity
- Give the model the current date with each command, so a date or date & time
  field gets a date instead of "Saturday"
- Edit the system and user prompts on the Settings page
- Reword the system prompt so the no-match reply is asked for in the requested
  language, rather than bundled into one sentence with what to say. This rebuilds
  the prompt cache once on upgrade

## 1.0.0

- Initial release
