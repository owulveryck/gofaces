package gofaces

import "sort"

type byProba []element

func (b byProba) Len() int           { return len(b) }
func (b byProba) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byProba) Less(i, j int) bool { return b[i].prob < b[j].prob }

type byConfidence []box

func (b byConfidence) Len() int           { return len(b) }
func (b byConfidence) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byConfidence) Less(i, j int) bool { return b[i].confidence < b[j].confidence }

func getOrderedElements(input []float64) []element {
	elems := make([]element, len(input))
	for i := 0; i < len(elems); i++ {
		elems[i] = element{
			prob:  input[i],
			class: classes[i],
		}
	}
	sort.Sort(sort.Reverse(byProba(elems)))
	return elems
}
