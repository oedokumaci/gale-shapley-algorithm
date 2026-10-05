import type { PersonImages } from '@/types';

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export const DEFAULT_PROPOSER_NAMES = [
  'Timothée Chalamet',
  'Tom Holland',
  'Jacob Elordi',
  'Pete Davidson',
  'Bad Bunny',
  'Kanye West',
];

export const DEFAULT_RESPONDER_NAMES = [
  'Kylie Jenner',
  'Zendaya',
  'Sabrina Carpenter',
  'Kendall Jenner',
  'Margot Robbie',
  'Kim Kardashian',
];

export function buildRandomPrefs(
  names: string[],
  otherSide: string[],
): Record<string, string[]> {
  const prefs: Record<string, string[]> = {};
  for (const name of names) {
    prefs[name] = shuffle(otherSide);
  }
  return prefs;
}

export const DEFAULT_IMAGES: PersonImages = {
  proposers: {
    'Timothée Chalamet':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Timoth%C3%A9e_Chalamet-63482_%28cropped%29.jpg/500px-Timoth%C3%A9e_Chalamet-63482_%28cropped%29.jpg',
    'Tom Holland':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Tom_Holland_during_pro-am_Wentworth_golf_club_2023-2.jpg/500px-Tom_Holland_during_pro-am_Wentworth_golf_club_2023-2.jpg',
    'Jacob Elordi':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/JacobElordi-TIFF2025-01_%28cropped_2%29.png/500px-JacobElordi-TIFF2025-01_%28cropped_2%29.png',
    'Pete Davidson':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Pete_Davidson_portrait_%28cropped%29.png/500px-Pete_Davidson_portrait_%28cropped%29.png',
    'Bad Bunny':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Bad_Bunny_2019_by_Glenn_Francis_%28cropped%29.jpg/500px-Bad_Bunny_2019_by_Glenn_Francis_%28cropped%29.jpg',
    'Kanye West':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Kanye_West_at_the_2009_Tribeca_Film_Festival_%28crop_2%29.jpg/500px-Kanye_West_at_the_2009_Tribeca_Film_Festival_%28crop_2%29.jpg',
  },
  responders: {
    'Kylie Jenner':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Kylie_Jenner1_%28cropped%29.png/500px-Kylie_Jenner1_%28cropped%29.png',
    'Zendaya':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Zendaya_-_2019_by_Glenn_Francis.jpg/500px-Zendaya_-_2019_by_Glenn_Francis.jpg',
    'Sabrina Carpenter':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Primavera2025_%28139_of_182%29_%2854574520207%29_%28cropped%29.jpg/500px-Primavera2025_%28139_of_182%29_%2854574520207%29_%28cropped%29.jpg',
    'Kendall Jenner':
      'https://upload.wikimedia.org/wikipedia/commons/5/54/Kendall_Jenner_for_Adanola_2_%28cropped%29.jpg',
    'Margot Robbie':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/SYDNEY%2C_AUSTRALIA_-_JANUARY_23_Margot_Robbie_arrives_at_the_Australian_Premiere_of_%27I%2C_Tonya%27_on_January_23%2C_2018_in_Sydney%2C_Australia_%2828074883999%29_%28cropped_2%29.jpg/500px-thumbnail.jpg',
    'Kim Kardashian':
      'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Kim_Kardashian_West_2014.jpg/500px-Kim_Kardashian_West_2014.jpg',
  },
};
