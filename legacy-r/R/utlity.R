
party_name <- function(name) {
  case_when(
    grepl('ANZ|Advance', name) ~ 'Advance NZ',
    grepl('ACT', name) ~ 'ACT',
    grepl('CON|^Con$|NCP', name) ~ 'New Conservative',
    grepl('GRN|Green', name) ~ 'Green',
    grepl('LAB|Labour', name) ~ 'Labour',
    grepl('MRI|TPM|Māori|Maori', name) ~ 'Te Pāti Māori',
    grepl('MNA|Mana', name) ~ 'Mana',
    grepl('NAT|National', name) ~ 'National',
    grepl('NZF|First', name) ~ 'NZ First',
    grepl('TOP', name) ~ 'TOP',
    grepl('UNF', name) ~ 'United Future',
    grepl('^Prog$', name) ~ 'Progressive',
    .default = name
  )
}
